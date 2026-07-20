"""Correctness gate: compare WavLM feature extraction across two devices.

Live-extracts the same utterances on two devices (not against a cached
file — cross-machine/cross-run cache staleness would defeat the point of
this check) and reports the diff distribution between them, plus the
effect on the trained probe's routing scores. No pass/fail threshold is
hardcoded: this is meant to be read by a human before any newly-extracted
device's cache is trusted.

Supports --batch-size for batched extraction (length-bucketed via
accentedness_routing.features.batching.bucket_by_duration, same
construction extract_features.py uses in production). Batching's own
masking correctness is verified separately, same-device, in
verify_batching_parity.py — this script's job is device parity, not
batching correctness, so keep batch_size fixed between two runs being
compared rather than changing both variables at once.

Usage:
    uv run python scripts/verify_features_device_parity.py \\
        --device-a cpu --device-b mps --num-samples 100

    # Batched:
    uv run python scripts/verify_features_device_parity.py \\
        --device-a cpu --device-b mps --batch-size 16

    # On a CUDA box, the same script, unchanged:
    uv run python scripts/verify_features_device_parity.py \\
        --device-a cpu --device-b cuda:0
    uv run python scripts/verify_features_device_parity.py \\
        --device-a cuda:0 --device-b cuda:1
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from accentedness_routing.features.batching import bucket_by_duration
from accentedness_routing.features.wavlm_extractor import WavLMExtractor
from accentedness_routing.triggers.scalar_probe import AccentednessProbe


def load_utterances(num_samples: int | None) -> list:
    data_dir = Path("data")
    all_utterances = []
    for split_name in ["train", "val", "test"]:
        pkl_path = data_dir / f"{split_name}_utterances.pkl"
        with open(pkl_path, "rb") as f:
            all_utterances.extend(pickle.load(f))
    if num_samples is not None and num_samples < len(all_utterances):
        # Evenly spaced sample (not just a prefix), so we don't
        # accidentally only cover one split.
        idx = np.linspace(0, len(all_utterances) - 1, num_samples).astype(int)
        all_utterances = [all_utterances[i] for i in idx]
    return all_utterances


def extract_all(
    extractor: WavLMExtractor,
    utterances: list,
    label: str,
    batch_size: int = 1,
    max_batch_duration: float | None = None,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, float, str]]]:
    """Extract features for all utterances, tolerating per-utterance/batch failures.

    A single pathological utterance (e.g. one long enough to blow up a
    device's O(T^2) attention-bias buffer) must not discard progress on
    the rest or abort a run that may have taken tens of minutes to reach
    that point. At batch_size>1, a failure takes down its whole batch
    (padding cost is shared across the batch), so every utterance in
    that batch is logged as failed, not just the culprit.

    Returns (features, failures) where failures is a list of
    (utterance_id, duration_seconds, error_message).
    """
    feats = {}
    failures: list[tuple[str, float, str]] = []
    t0 = time.time()

    if batch_size == 1:
        for i, utt in enumerate(utterances):
            dur = len(utt.audio) / utt.sample_rate
            try:
                feats[utt.utterance_id] = extractor.extract(utt.audio, utt.sample_rate)
            except RuntimeError as e:
                print(f"  [{label}] FAILED {utt.utterance_id} (duration {dur:.1f}s): {e}")
                failures.append((utt.utterance_id, dur, str(e)))
            if (i + 1) % 200 == 0:
                print(f"  [{label}] {i+1}/{len(utterances)} ({time.time()-t0:.1f}s)")
    else:
        batches = bucket_by_duration(utterances, batch_size, max_batch_duration)
        n_done = 0
        for batch in batches:
            try:
                results = extractor.extract_batch([u.audio for u in batch], batch[0].sample_rate)
                for utt, feat in zip(batch, results):
                    feats[utt.utterance_id] = feat
            except RuntimeError as e:
                durs = [len(u.audio) / u.sample_rate for u in batch]
                print(f"  [{label}] FAILED batch of {len(batch)} "
                      f"(durations {min(durs):.1f}-{max(durs):.1f}s): {e}")
                for utt, dur in zip(batch, durs):
                    failures.append((utt.utterance_id, dur, str(e)))
            n_done += len(batch)
            print(f"  [{label}] {n_done}/{len(utterances)} ({time.time()-t0:.1f}s)")

    print(
        f"  [{label}] done: {len(feats)}/{len(utterances)} succeeded, "
        f"{len(failures)} failed, in {time.time()-t0:.1f}s"
    )
    return feats, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device-a", default="cpu", help="Reference device")
    parser.add_argument("--device-b", required=True, help="Device to validate against A")
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Subsample size (default: all utterances across train/val/test)",
    )
    parser.add_argument(
        "--probe-path", default="models/probe.pt",
        help="Trained probe checkpoint to check downstream score parity (skip if absent)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for extraction on both devices (length-bucketed if >1)",
    )
    parser.add_argument(
        "--max-batch-duration", type=float, default=None,
        help="Shrink batch size for utterances longer than this (seconds) to avoid "
             "the O(T^2) attention-bias OOM cliff. Device/memory-specific; default "
             "None means fixed batch_size regardless of duration.",
    )
    parser.add_argument(
        "--exclude-ids", default="",
        help="Comma-separated utterance IDs to exclude before running (e.g. known "
             "extreme-duration outliers that risk an uncatchable OOM when batched "
             "with other long utterances, rather than a catchable RuntimeError). "
             "Empty by default — a no-op unless explicitly passed.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["features"]["model"]
    num_layers = cfg["features"]["num_layers"]
    hidden_dim = cfg["features"]["hidden_dim"]

    utterances = load_utterances(args.num_samples)

    exclude_ids = {x for x in args.exclude_ids.split(",") if x}
    if exclude_ids:
        n_before = len(utterances)
        utterances = [u for u in utterances if u.utterance_id not in exclude_ids]
        print(f"Excluded {n_before - len(utterances)} utterances via --exclude-ids: {exclude_ids}")

    n_total = len(utterances)
    print(f"Comparing {n_total} utterances: device A={args.device_a!r} vs device B={args.device_b!r}, "
          f"batch_size={args.batch_size}, max_batch_duration={args.max_batch_duration}\n")

    print(f"Loading extractor on {args.device_a}...")
    extractor_a = WavLMExtractor(model_name, args.device_a)
    t0 = time.time()
    feats_a, failures_a = extract_all(
        extractor_a, utterances, args.device_a, args.batch_size, args.max_batch_duration
    )
    time_a = time.time() - t0
    extractor_a.cleanup()

    print(f"\nLoading extractor on {args.device_b}...")
    extractor_b = WavLMExtractor(model_name, args.device_b)
    t0 = time.time()
    feats_b, failures_b = extract_all(
        extractor_b, utterances, args.device_b, args.batch_size, args.max_batch_duration
    )
    time_b = time.time() - t0
    extractor_b.cleanup()

    print(f"\n=== Timing ===\n  {args.device_a}: {time_a:.1f}s\n  {args.device_b}: {time_b:.1f}s")

    if failures_a or failures_b:
        print(f"\n=== Extraction failures ({len(failures_a)} on {args.device_a}, "
              f"{len(failures_b)} on {args.device_b}) ===")
        for uid, dur, err in failures_a:
            print(f"  [{args.device_a}] {uid} (duration {dur:.1f}s): {err[:120]}")
        for uid, dur, err in failures_b:
            print(f"  [{args.device_b}] {uid} (duration {dur:.1f}s): {err[:120]}")
        print("Excluding these utterances from the diff comparison below "
              "(compared only where both devices succeeded).")

    # Only compare utterances that succeeded on both devices.
    common_ids = set(feats_a) & set(feats_b)
    utterances = [u for u in utterances if u.utterance_id in common_ids]
    print(f"\nComparing {len(utterances)}/{n_total} utterances that succeeded on both devices.")

    # ------------------------------------------------------------------
    # Per-utterance, per-layer diff stats
    # ------------------------------------------------------------------
    uids = [u.utterance_id for u in utterances]
    durations = np.array([len(u.audio) / u.sample_rate for u in utterances])
    max_abs = np.zeros((len(uids), num_layers))
    mean_abs = np.zeros((len(uids), num_layers))
    rel_abs = np.zeros((len(uids), num_layers))  # max_abs_diff / max|a| for that layer, scale-normalized
    cos_sim = np.zeros((len(uids), num_layers))

    for i, uid in enumerate(uids):
        a, b = feats_a[uid], feats_b[uid]
        diff = (a - b).abs()
        layer_max = diff.amax(dim=1)
        max_abs[i] = layer_max.numpy()
        mean_abs[i] = diff.mean(dim=1).numpy()
        scale = a.abs().amax(dim=1).clamp_min(1e-8)
        rel_abs[i] = (layer_max / scale).numpy()
        cos_sim[i] = torch.nn.functional.cosine_similarity(a, b, dim=1).numpy()

    print("\n=== Feature parity: device A vs device B ===")
    print(f"Global max abs diff: {max_abs.max():.6e} "
          f"(utterance {uids[np.unravel_index(max_abs.argmax(), max_abs.shape)[0]]}, "
          f"layer {np.unravel_index(max_abs.argmax(), max_abs.shape)[1]})")
    print(f"Global mean abs diff: {mean_abs.mean():.6e}")
    per_utt_max = max_abs.max(axis=1)
    print(f"Per-utterance max-abs-diff percentiles: "
          f"p50={np.percentile(per_utt_max, 50):.6e}  "
          f"p95={np.percentile(per_utt_max, 95):.6e}  "
          f"p99={np.percentile(per_utt_max, 99):.6e}  "
          f"max={per_utt_max.max():.6e}")
    print(f"Cosine similarity: min={cos_sim.min():.8f}  mean={cos_sim.mean():.8f}")
    per_utt_rel = rel_abs.max(axis=1)
    print(f"Per-utterance relative diff (max_abs_diff / max|activation|) percentiles: "
          f"p50={np.percentile(per_utt_rel, 50):.4%}  "
          f"p95={np.percentile(per_utt_rel, 95):.4%}  "
          f"p99={np.percentile(per_utt_rel, 99):.4%}  "
          f"max={per_utt_rel.max():.4%}")

    print("\nPer-layer mean(max-abs-diff) and mean(relative diff) across utterances "
          "(layer 0 = input embeddings):")
    for layer in range(num_layers):
        print(f"  layer {layer:2d}: mean_max_abs_diff={max_abs[:, layer].mean():.6e}  "
              f"mean_rel_diff={rel_abs[:, layer].mean():.4%}  "
              f"cos_sim={cos_sim[:, layer].mean():.8f}")

    # ------------------------------------------------------------------
    # Duration-stratified breakdown — short utterances have few time
    # frames to mean-pool over, so per-kernel numerical noise averages
    # out less. Check whether tolerance holds in that regime specifically.
    # ------------------------------------------------------------------
    buckets = [(0.0, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, np.inf)]
    print("\nDuration-stratified relative diff (per-utterance max over layers):")
    for lo, hi in buckets:
        mask = (durations >= lo) & (durations < hi)
        n = mask.sum()
        if n == 0:
            continue
        bucket_rel = per_utt_rel[mask]
        bucket_cos = cos_sim[mask].min(axis=1)
        label = f"[{lo:.1f}s, {hi:.1f}s)" if hi != np.inf else f"[{lo:.1f}s, inf)"
        print(f"  {label:>16}  n={n:4d}  rel_diff p50={np.percentile(bucket_rel,50):.4%}  "
              f"p99={np.percentile(bucket_rel,99):.4%}  max={bucket_rel.max():.4%}  "
              f"min_cos_sim={bucket_cos.min():.6f}")

    # ------------------------------------------------------------------
    # Downstream: does drift change the probe's routing scores?
    # ------------------------------------------------------------------
    probe_path = Path(args.probe_path)
    if probe_path.exists():
        checkpoint = torch.load(probe_path, weights_only=False)
        probe = AccentednessProbe(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            probe_dim=checkpoint["config"]["hidden_dim"],
            dropout=checkpoint["config"]["dropout"],
        )
        probe.load_state_dict(checkpoint["model_state_dict"])
        probe.eval()
        cal = checkpoint["calibration"]

        def calibrate(raw: float) -> float:
            low, high = cal["low"], cal["high"]
            rng = high - low
            if rng < 1e-8:
                return 0.5
            return max(0.0, min(1.0, (raw - low) / rng))

        with torch.no_grad():
            scores_a = np.array([calibrate(probe(feats_a[uid].unsqueeze(0)).item()) for uid in uids])
            scores_b = np.array([calibrate(probe(feats_b[uid].unsqueeze(0)).item()) for uid in uids])

        score_diff = np.abs(scores_a - scores_b)
        from scipy.stats import spearmanr
        rho = spearmanr(scores_a, scores_b).statistic

        print("\n=== Downstream: probe routing-score parity ===")
        print(f"Max abs score diff: {score_diff.max():.6e}")
        print(f"Mean abs score diff: {score_diff.mean():.6e}")
        print(f"Spearman rank correlation (scores A vs B): {rho:.8f}")
    else:
        print(f"\nNo probe checkpoint at {probe_path}, skipping downstream score check.")


if __name__ == "__main__":
    main()
