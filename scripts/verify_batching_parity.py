"""Correctness gate: batched vs unbatched WavLM extraction, same device.

Isolates "does batching (padding + masked pooling) introduce drift" from
device drift entirely — everything here runs on one device. A masking
bug has a specific signature: diffs correlated with how much padding an
utterance sat next to (i.e. how much shorter it was than the longest
utterance in its batch), not uniform noise. This script is built to
surface exactly that correlation, not just an aggregate pass/fail.

Tests both extract_batch (mean-pooled) and extract_stats_batch
(mean+std) — the masked-variance path in the latter is the more likely
place for a low-valid-frame bug to surface, since variance needs at
least 2 valid frames to be well-defined.

Usage:
    uv run python scripts/verify_batching_parity.py --device cpu --batch-size 16
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr

from accentedness_routing.features.batching import bucket_by_duration
from accentedness_routing.features.wavlm_extractor import WavLMExtractor


def load_utterances() -> list:
    data_dir = Path("data")
    all_utterances = []
    for split_name in ["train", "val", "test"]:
        with open(data_dir / f"{split_name}_utterances.pkl", "rb") as f:
            all_utterances.extend(pickle.load(f))
    return all_utterances


def relative_diff(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per-layer max-abs-diff normalized by that layer's own activation scale."""
    diff = (a - b).abs().amax(dim=1)
    scale = a.abs().amax(dim=1).clamp_min(1e-8)
    return (diff / scale).numpy()


def run_check(
    extractor: WavLMExtractor,
    utterances: list,
    batch_size: int,
    unbatched_fn_name: str,
    batched_fn_name: str,
    label: str,
    max_batch_duration: float | None = None,
):
    unbatched_fn = getattr(extractor, unbatched_fn_name)
    batched_fn = getattr(extractor, batched_fn_name)

    durations = np.array([len(u.audio) / u.sample_rate for u in utterances])
    uids = [u.utterance_id for u in utterances]

    # --- Unbatched reference ---
    print(f"\n[{label}] Computing unbatched reference ({unbatched_fn_name})...")
    t0 = time.time()
    ref = {}
    for i, utt in enumerate(utterances):
        ref[utt.utterance_id] = unbatched_fn(utt.audio, utt.sample_rate)
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(utterances)} ({time.time()-t0:.1f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    # --- Batched, length-bucketed (sorted by duration before chunking),
    # matching the real extract_features.py batch construction exactly ---
    print(f"[{label}] Computing batched ({batched_fn_name}, batch_size={batch_size})...")
    t0 = time.time()
    batched = {}
    is_longest_in_batch = {}
    relative_padding = {}  # 1 - own_duration / batch_max_duration
    batch_idx_of = {}
    batches = bucket_by_duration(utterances, batch_size, max_batch_duration)
    for b, batch in enumerate(batches):
        batch_durs = np.array([len(u.audio) / u.sample_rate for u in batch])
        batch_max = batch_durs.max()
        results = batched_fn([u.audio for u in batch], batch[0].sample_rate)
        for utt, feat, dur in zip(batch, results, batch_durs):
            batched[utt.utterance_id] = feat
            is_longest_in_batch[utt.utterance_id] = bool(dur == batch_max)
            relative_padding[utt.utterance_id] = float(1.0 - dur / batch_max) if batch_max > 0 else 0.0
            batch_idx_of[utt.utterance_id] = b
    print(f"  done in {time.time()-t0:.1f}s, {len(batches)} batches")

    # --- Diff ---
    rel_diff_max = np.zeros(len(uids))  # max over layers, per utterance
    for j, uid in enumerate(uids):
        rd = relative_diff(ref[uid], batched[uid])
        rel_diff_max[j] = rd.max()

    pad_frac = np.array([relative_padding[uid] for uid in uids])
    is_longest = np.array([is_longest_in_batch[uid] for uid in uids])

    print(f"\n=== [{label}] Aggregate ===")
    print(f"Relative diff percentiles: p50={np.percentile(rel_diff_max,50):.4%} "
          f"p95={np.percentile(rel_diff_max,95):.4%} p99={np.percentile(rel_diff_max,99):.4%} "
          f"max={rel_diff_max.max():.4%}")

    print(f"\n=== [{label}] Stratified by longest-in-batch ===")
    for flag, name in [(True, "longest-in-batch (no padding)"), (False, "padded (shorter than batch max)")]:
        mask = is_longest == flag
        n = mask.sum()
        if n == 0:
            continue
        vals = rel_diff_max[mask]
        print(f"  {name:38s} n={n:4d}  p50={np.percentile(vals,50):.4%}  "
              f"p99={np.percentile(vals,99):.4%}  max={vals.max():.4%}")

    print(f"\n=== [{label}] Correlation: relative diff vs. how much padding an utterance sat next to ===")
    # A masking bug's signature: positive correlation. Diffs among
    # never-padded (relative_padding==0) utterances isolate device/batching
    # numerical noise with the masking variable held at zero.
    rho, pval = pearsonr(pad_frac, rel_diff_max)
    print(f"  Pearson r(relative_padding, relative_diff) = {rho:.4f} (p={pval:.4g})")
    zero_pad_mask = pad_frac == 0.0
    if zero_pad_mask.sum() > 0:
        print(f"  Among {zero_pad_mask.sum()} utterances with ZERO relative padding: "
              f"p50={np.percentile(rel_diff_max[zero_pad_mask],50):.4%} "
              f"max={rel_diff_max[zero_pad_mask].max():.4%}")

    print(f"\n=== [{label}] Duration-stratified ===")
    buckets = [(0.0, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, np.inf)]
    for lo, hi in buckets:
        mask = (durations >= lo) & (durations < hi)
        n = mask.sum()
        if n == 0:
            continue
        vals = rel_diff_max[mask]
        label_b = f"[{lo:.1f}s, {hi:.1f}s)" if hi != np.inf else f"[{lo:.1f}s, inf)"
        print(f"  {label_b:>16}  n={n:4d}  p50={np.percentile(vals,50):.4%}  "
              f"p99={np.percentile(vals,99):.4%}  max={vals.max():.4%}")

    print(f"\n=== [{label}] Shortest utterances individually (not averaged into a bucket) ===")
    shortest_idx = np.argsort(durations)[:15]
    for j in shortest_idx:
        uid = uids[j]
        print(f"  {uid}  dur={durations[j]:.3f}s  batch={batch_idx_of[uid]:4d}  "
              f"longest_in_batch={is_longest_in_batch[uid]!s:5s}  "
              f"rel_padding={relative_padding[uid]:.2%}  rel_diff={rel_diff_max[j]:.4%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-batch-duration", type=float, default=None,
        help="Shrink batch size for utterances longer than this (seconds). "
             "Device/memory-specific; default None means fixed batch_size.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["features"]["model"]

    utterances = load_utterances()

    # The 3 extreme-duration outliers (536.0s, 356.6s, 199.5s) already
    # crashed MPS unbatched on the attention-bias O(T^2) buffer (see
    # verify_features_device_parity.py). Batching one of them at
    # batch_size>1 multiplies that same cost by the batch dimension —
    # risking a genuine CPU memory crash here, which would be testing that
    # known, separately-tracked limitation, not the masking logic this
    # script exists to check. Excluded deliberately, not silently.
    EXTREME_DURATION_OUTLIERS = {"22b35c5c8035", "bf712712d64b", "2ca79ae31da7"}
    n_before = len(utterances)
    utterances = [u for u in utterances if u.utterance_id not in EXTREME_DURATION_OUTLIERS]
    print(f"Excluded {n_before - len(utterances)} extreme-duration outliers "
          f"(id in {EXTREME_DURATION_OUTLIERS}): known O(T^2) attention-bias issue, "
          f"tracked separately.")

    print(f"Loaded {len(utterances)} utterances. Device={args.device}, batch_size={args.batch_size}")

    extractor = WavLMExtractor(model_name, args.device)

    run_check(
        extractor, utterances, args.batch_size,
        "extract", "extract_batch", "mean-pooled",
        max_batch_duration=args.max_batch_duration,
    )
    run_check(
        extractor, utterances, args.batch_size,
        "extract_stats", "extract_stats_batch", "mean+std",
        max_batch_duration=args.max_batch_duration,
    )

    extractor.cleanup()


if __name__ == "__main__":
    main()
