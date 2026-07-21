"""EXP-13 DDP diagnostic: per-rank trainable-parameter breakdown, construction only.

Isolates model construction + freeze from everything else in train_ddp.py (no data
loading, no training loop) to capture the exact state DDP's constructor checks
(_build_params_for_reducer in torch/nn/parallel/distributed.py counts parameters with
requires_grad=True) on every rank, printed *before* the DDP() wrap -- so the data is
captured even if the wrap itself hangs or raises.

Background: a real 2x A40 torchrun run of scripts/train_ddp.py failed at the DDP()
constructor with "Rank 0 has 472 params, while rank 1 has inconsistent 0 params".
Every code path that could plausibly cause this (rank-conditioned freezing, a
local_rank truthiness bug, FSDP/meta-device loading, concurrent from_pretrained
cache races, transformers version drift) has been checked and ruled out on CPU/gloo
-- both ranks build 472/472 matching trainable params there, every time. The one
untested axis is real CUDA + nccl with per-rank device placement (cuda:0 vs cuda:1),
which cannot be exercised on a CPU-only machine. Run this on the next GPU box,
first, before any training, to capture what actually diverges.

Launch (2 GPUs, real repro conditions):
    torchrun --nproc_per_node=2 scripts/diag_ddp_params.py --config configs/default.yaml

Also runs single-process or CPU/gloo (auto-fallback, same as train_ddp.py's
setup_ddp) for a quick sanity check that the script itself is correct -- that
path is NOT expected to show the bug; it's only there to validate the diagnostic
before trusting it on rented hardware.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from accentedness_routing.triggers.unfrozen_wavlm_probe import WavLMGainProbe


def setup_ddp(backend: str) -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
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


def trainable_count(model: torch.nn.Module) -> int:
    return sum(1 for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ecfg = cfg["exp13"]

    rank, local_rank, world_size, device = setup_ddp(ecfg["backend"])
    tag = f"[rank={rank} local_rank={local_rank} device={device}]"
    print(f"{tag} world_size={world_size} starting construction", flush=True)

    freeze_encoder = ecfg.get("freeze_encoder", False)
    freeze_feature_extractor = ecfg.get("freeze_feature_extractor", True)
    model = WavLMGainProbe(
        model_name=ecfg["model"],
        num_layers=ecfg["num_layers"],
        hidden_dim=ecfg["hidden_dim"],
        probe_dim=ecfg["probe_hidden_dim"],
        dropout=ecfg["dropout"],
        freeze_feature_extractor=freeze_feature_extractor,
        freeze_encoder=freeze_encoder,
        gradient_checkpointing=ecfg.get("gradient_checkpointing", False),
        disable_train_augmentation=ecfg.get("disable_train_augmentation", True),
    ).to(device)

    total = sum(1 for _ in model.parameters())
    trainable = trainable_count(model)

    # One representative param from each of the three submodules the freeze logic
    # can affect differently: the conv frontend (frozen iff freeze_feature_extractor),
    # a transformer layer (frozen iff freeze_encoder), and the probe head (never
    # frozen by any code path -- if this one is False, freezing has leaked somewhere
    # it shouldn't).
    fe_param = next(model.wavlm.feature_extractor.parameters())
    enc_param = next(model.wavlm.encoder.layers[0].parameters())
    probe_param = next(model.probe.parameters())

    print(
        f"{tag} total_params={total} trainable_params={trainable} "
        f"freeze_encoder={freeze_encoder} freeze_feature_extractor={freeze_feature_extractor} "
        f"| feature_extractor.requires_grad={fe_param.requires_grad} "
        f"| encoder.layers[0].requires_grad={enc_param.requires_grad} "
        f"| probe.requires_grad={probe_param.requires_grad} "
        f"| param_device={fe_param.device}",
        flush=True,
    )

    if world_size > 1:
        dist.barrier()  # let every rank's print land before anyone attempts the wrap
        print(f"{tag} attempting DDP() wrap...", flush=True)
        try:
            model = DDP(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,
            )
            print(f"{tag} DDP() wrap OK", flush=True)
        except RuntimeError as e:
            print(f"{tag} DDP() wrap FAILED: {e}", flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
