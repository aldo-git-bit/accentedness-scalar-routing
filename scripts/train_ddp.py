"""EXP-13: DDP fine-tuning of WavLM-large (unfrozen transformer) + probe head
on the capped escalation-gain target.

Target hardware: 2x A40, single node. The exact same script runs the CPU
smoke test on a laptop and the real DDP run on the GPU box — device,
precision, and world size are all read from the environment / CLI flags,
never hardcoded.

Launch:
    # CPU smoke test — no torchrun, single process, catches wiring bugs
    # (import errors, shape mismatches, a silently-broken gradient path)
    # for free before renting anything:
    uv run python scripts/train_ddp.py --config configs/default.yaml --smoke

    # Real DDP run, 2 GPUs, single node:
    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/default.yaml

    # Resume from the last checkpoint:
    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/default.yaml --resume
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from scipy.stats import pearsonr
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Wav2Vec2FeatureExtractor

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.data.audio_dataset import AudioGainDataset, Collator
from accentedness_routing.eval.eval_common import escalation_gain
from accentedness_routing.features.batching import bucket_by_duration
from accentedness_routing.triggers.unfrozen_wavlm_probe import WavLMGainProbe


# ---------------------------------------------------------------------------
# DDP setup — reads torchrun's env vars (RANK, LOCAL_RANK, WORLD_SIZE), no
# manual rendezvous needed since we're single-node only. Falls back to a
# single, unwrapped process when launched directly (python, not torchrun) —
# that fallback path is exactly what makes --smoke work without renting.
# ---------------------------------------------------------------------------

def setup_ddp(backend: str) -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # NCCL requires CUDA; fall back to gloo if launched under torchrun
        # without GPUs (e.g. a multi-process CPU smoke test).
        actual_backend = backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=actual_backend)
    else:
        rank, local_rank, world_size = 0, 0, 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int):
    # Every rank calls this with the identical seed — critical for
    # DistributedSampler(seed=seed) to compute the same shuffle permutation
    # on every rank before each rank takes its own disjoint slice (see
    # DurationBucketBatchSampler below). DDP's constructor also broadcasts
    # rank 0's initial weights to every other rank regardless, so identical
    # seeding isn't what makes model init match across ranks — it's what
    # makes the *data partition* correct.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split_with_gain_targets(split_name: str, cfg: dict) -> tuple[list, dict]:
    """Mirrors train_probe_ext1.py's load_split_data, but keeps raw audio
    (needed for the trainable encoder) instead of precomputed features."""
    data_dir = Path("data")
    with open(data_dir / f"{split_name}_utterances.pkl", "rb") as f:
        utterances = pickle.load(f)

    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    kept, targets = [], {}
    for utt in utterances:
        d = load_cached(cache_dir, default_model, utt.utterance_id)
        c = load_cached(cache_dir, careful_model, utt.utterance_id)
        if d is None or c is None:
            continue
        targets[utt.utterance_id] = escalation_gain(d["wer"], c["wer"])
        kept.append(utt)
    return kept, targets


class DurationBucketBatchSampler:
    """Composes DistributedSampler (which utterances this rank owns, and in
    what shuffled order per epoch) with bucket_by_duration (Stage 2,
    unmodified — how those utterances get grouped into batches). These are
    deliberately separate, composable concerns: sharding decides ownership,
    bucketing decides grouping, same principle as extraction's
    --shard-index/--num-shards composing cleanly with bucket_by_duration.

    Standard PyTorch batch_sampler contract (yields lists of dataset
    indices), so this still works with DataLoader's num_workers>0 worker
    processes even though batch membership depends on audio duration, not
    just position.
    """

    def __init__(
        self,
        dataset: AudioGainDataset,
        sampler: DistributedSampler,
        batch_size: int,
        max_batch_duration: float | None,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.max_batch_duration = max_batch_duration
        self._id_to_index = {u.utterance_id: i for i, u in enumerate(dataset.utterances)}

    def __iter__(self):
        indices = list(self.sampler)  # this rank's shuffled slice for the current epoch
        rank_utterances = [self.dataset.utterances[i] for i in indices]
        batches = bucket_by_duration(rank_utterances, self.batch_size, self.max_batch_duration)
        for batch in batches:
            yield [self._id_to_index[u.utterance_id] for u in batch]

    def __len__(self) -> int:
        # Approximate — exact batch count varies with duration-driven batch
        # sizing. Only used for progress-bar totals, not correctness.
        return max(1, (len(self.sampler) + self.batch_size - 1) // self.batch_size)


# ---------------------------------------------------------------------------
# Checkpointing — rank 0 only. Strips DDP's "module." prefix on save so the
# checkpoint loads cleanly into a bare (non-DDP-wrapped) model later, e.g.
# for evaluation. Saves RNG state for exact resumability; step_in_epoch lets
# --resume skip already-seen batches (iteration cost only, no wasted
# forward/backward) rather than only resuming at epoch boundaries.
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    epoch: int, step_in_epoch: int, cfg: dict,
):
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": cfg,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    torch.set_rng_state(ckpt["rng_state"].cpu())
    if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [s.cpu().to(torch.uint8) for s in ckpt["cuda_rng_state"]]
        )
    return ckpt["epoch"], ckpt["step_in_epoch"]


# ---------------------------------------------------------------------------
# Evaluation — rank 0 only, full (unsharded) val set. Simpler than
# aggregating partial-val metrics across ranks via all-reduce, and val runs
# infrequently enough (once per epoch) that this isn't a bottleneck.
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module, val_utterances: list, val_targets: dict,
    processor, device: torch.device, criterion: nn.Module,
) -> tuple[float, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for utt in val_utterances:
            inputs = processor(utt.audio, sampling_rate=utt.sample_rate, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
            pred = raw_model(input_values, attention_mask)
            preds.append(pred.item())
            targets.append(val_targets[utt.utterance_id])

    preds_t = torch.tensor(preds, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)
    val_loss = criterion(preds_t, targets_t).item()
    r = pearsonr(preds, targets).statistic if len(preds) > 1 else 0.0
    return val_loss, float(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true",
                         help="2 batches, CPU-friendly, single process. Catches wiring "
                              "errors (imports, shapes, a silently-broken gradient path) "
                              "before renting anything.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ecfg = cfg["exp13"]
    seed = cfg.get("seed", 42)

    rank, local_rank, world_size, device = setup_ddp(ecfg["backend"])
    seed_everything(seed)  # identical on every rank — see seed_everything's docstring

    if is_main_process(rank):
        print(f"world_size={world_size} device={device} smoke={args.smoke}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(ecfg["model"])

    train_utts, train_targets = load_split_with_gain_targets("train", cfg)
    val_utts, val_targets = load_split_with_gain_targets("val", cfg)
    if is_main_process(rank):
        print(f"train={len(train_utts)} val={len(val_utts)} (with cached ASR WER on both models)")

    train_dataset = AudioGainDataset(train_utts, train_targets)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    batch_sampler = DurationBucketBatchSampler(
        train_dataset, sampler, ecfg["batch_size"], ecfg.get("max_batch_duration")
    )
    collator = Collator(processor)
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, collate_fn=collator,
        num_workers=ecfg.get("num_workers", 0),
    )

    freeze_encoder = ecfg.get("freeze_encoder", False)
    model = WavLMGainProbe(
        model_name=ecfg["model"],
        num_layers=ecfg["num_layers"],
        hidden_dim=ecfg["hidden_dim"],
        probe_dim=ecfg["probe_hidden_dim"],
        dropout=ecfg["dropout"],
        freeze_feature_extractor=ecfg.get("freeze_feature_extractor", True),
        freeze_encoder=freeze_encoder,
        gradient_checkpointing=ecfg.get("gradient_checkpointing", False),
        disable_train_augmentation=ecfg.get("disable_train_augmentation", True),
    ).to(device)

    if world_size > 1:
        # frozen (requires_grad=False) params are excluded from DDP's
        # gradient-sync tracking entirely, so freeze_encoder alone doesn't
        # require this. Set True unconditionally instead: the risk this
        # guards against (a requires_grad=True parameter that goes unused
        # in some forward pass, e.g. from dynamic control flow inside
        # WavLM's architecture that hasn't been fully audited) causes a
        # hard DDP hang if wrong, not just a slowdown — worth the small
        # perf cost of the extra bookkeeping until profiled otherwise.
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=ecfg["lr"], weight_decay=ecfg["weight_decay"])
    criterion = nn.HuberLoss(delta=ecfg["huber_delta"])

    use_amp = ecfg.get("precision", "bf16") == "bf16" and device.type == "cuda"

    start_epoch, start_step = 0, 0
    model_path = Path(ecfg["model_path"])
    if args.resume and model_path.exists():
        start_epoch, start_step = load_checkpoint(model_path, model, optimizer, device)
        if is_main_process(rank):
            print(f"Resumed from epoch={start_epoch} step_in_epoch={start_step}")

    # -----------------------------------------------------------------
    # --smoke: 2 batches only, with the gradient-flow assertion. This is
    # the single most important guard from the plan — a no_grad/detach
    # bug wouldn't crash, it would just silently train a model where the
    # encoder never updates. Checked explicitly here rather than trusted.
    # -----------------------------------------------------------------
    if args.smoke:
        sampler.set_epoch(0)
        model.train()
        n_done = 0
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                preds = model(input_values, attention_mask).squeeze(-1)
                loss = criterion(preds, targets)
            loss.backward()

            if not freeze_encoder:
                raw_model = model.module if isinstance(model, DDP) else model
                wavlm_params_with_grad = [
                    p for p in raw_model.wavlm.parameters() if p.requires_grad and p.grad is not None
                ]
                assert wavlm_params_with_grad, "No WavLM parameter received a gradient — check for a stray no_grad/detach."
                nonzero = any(torch.count_nonzero(p.grad) > 0 for p in wavlm_params_with_grad)
                assert nonzero, "Every WavLM gradient is exactly zero — training would silently do nothing to the encoder."
                if is_main_process(rank):
                    print(f"  gradient-flow check passed: {len(wavlm_params_with_grad)} WavLM tensors received nonzero gradients")

            optimizer.step()
            n_done += 1
            if is_main_process(rank):
                print(f"  smoke batch {n_done}: loss={loss.item():.4f} batch_size={input_values.shape[0]}")
            if n_done >= 2:
                break

        # Checkpoint save/load round-trip.
        if is_main_process(rank):
            model_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model_path, model, optimizer, epoch=0, step_in_epoch=n_done, cfg=cfg)
            load_checkpoint(model_path, model, optimizer, device)
            print(f"  checkpoint save/load round-trip OK: {model_path}")
            print("SMOKE TEST PASSED")

        if world_size > 1:
            # Every rank must reach this point before any rank tears down
            # the process group — without it, a non-zero rank (which skips
            # the block above entirely) could call cleanup_ddp() while rank
            # 0 is still mid-checkpoint-I/O.
            dist.barrier()
        cleanup_ddp()
        return

    # -----------------------------------------------------------------
    # Real training loop
    # -----------------------------------------------------------------
    checkpoint_interval = ecfg.get("checkpoint_interval_steps", 50)
    for epoch in range(start_epoch, ecfg["max_epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                continue  # mid-epoch resume: skip already-seen batches (iteration only, no compute)

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                preds = model(input_values, attention_mask).squeeze(-1)
                loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            if is_main_process(rank) and step % checkpoint_interval == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} ({time.time()-t0:.1f}s)")

            if step % checkpoint_interval == 0:
                if is_main_process(rank):
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(model_path, model, optimizer, epoch, step + 1, cfg)
                if world_size > 1:
                    dist.barrier()  # other ranks wait until rank 0's write completes

        if is_main_process(rank):
            val_loss, val_r = evaluate(model, val_utts, val_targets, processor, device, criterion)
            print(f"epoch={epoch} done  val_loss={val_loss:.4f}  val_pearson_r={val_r:.4f}  ({time.time()-t0:.1f}s)")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model_path, model, optimizer, epoch + 1, 0, cfg)
        if world_size > 1:
            dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()
