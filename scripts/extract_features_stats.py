"""Extension 4: Extract mean+std WavLM features for all utterances.

Saves to data/features_cache_stats/. Shape (25, 2048) per utterance.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from accentedness_routing.features.batching import bucket_by_duration
from accentedness_routing.features.wavlm_extractor import WavLMExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract mean+std WavLM features for stats pooling")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--device", default=None,
        help="Override features.device from config (e.g. cuda:0, cuda:1, mps, cpu)",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="This process's shard index (0-based), for data-parallel extraction across devices",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards; utterances are assigned by index % num_shards",
    )
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"shard-index {args.shard_index} must be in [0, {args.num_shards})")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["features_stats"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["features"]["model"]
    device = args.device if args.device is not None else cfg["features"]["device"]

    # Load all utterances (stable order: train, then val, then test, as
    # pickled) so that sharding by index is identical across processes.
    data_dir = Path("data")
    all_utterances = []
    for split_name in ["train", "val", "test"]:
        pkl_path = data_dir / f"{split_name}_utterances.pkl"
        if not pkl_path.exists():
            print(f"Warning: {pkl_path} not found, skipping")
            continue
        with open(pkl_path, "rb") as f:
            utts = pickle.load(f)
        all_utterances.extend(utts)
        print(f"Loaded {len(utts)} utterances from {split_name}")

    print(f"\nTotal: {len(all_utterances)} utterances")

    if args.num_shards > 1:
        all_utterances = [
            u for i, u in enumerate(all_utterances) if i % args.num_shards == args.shard_index
        ]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"{len(all_utterances)} utterances assigned to this process"
        )

    print(f"Loading {model_name} on {device}...")
    extractor = WavLMExtractor(model_name, device)

    batch_size = cfg["features"].get("batch_size", 1)
    max_batch_duration = cfg["features"].get("max_batch_duration")

    n_cached = 0
    to_process = []
    for utt in all_utterances:
        out_path = cache_dir / f"{utt.utterance_id}.pt"
        if out_path.exists():
            n_cached += 1
        else:
            to_process.append(utt)

    print(f"{len(to_process)} utterances to extract, {n_cached} already cached "
          f"(batch_size={batch_size}, max_batch_duration={max_batch_duration})")

    # Bucketing happens on to_process, which is already restricted to this
    # shard's slice (sharding above) and to not-yet-cached utterances (just
    # above) — so it composes with sharding rather than crossing shard
    # boundaries, and never changes which cache file an utterance lands in
    # (still keyed by utterance_id below, regardless of processing order).
    batches = bucket_by_duration(to_process, batch_size, max_batch_duration) if batch_size > 1 else [
        [u] for u in to_process
    ]

    n_extracted = 0
    n_failed = 0
    pbar = tqdm(total=len(to_process), desc="Extracting stats features")
    for batch in batches:
        try:
            if batch_size == 1:
                results = [extractor.extract_stats(batch[0].audio, batch[0].sample_rate)]
            else:
                results = extractor.extract_stats_batch(
                    [u.audio for u in batch], batch[0].sample_rate
                )
            for utt, features in zip(batch, results):
                torch.save(features, cache_dir / f"{utt.utterance_id}.pt")
                n_extracted += 1
        except (ValueError, RuntimeError) as e:
            # A batch failure (e.g. one pathological long utterance OOMing
            # the padded forward pass) skips every utterance in that batch,
            # not just the culprit — logged so it's diagnosable which ones.
            ids = [u.utterance_id for u in batch]
            print(f"\n  SKIP batch {ids}: {e}")
            n_failed += len(batch)
        pbar.update(len(batch))
    pbar.close()

    print(f"\nDone: {n_extracted} extracted, {n_cached} cached, {n_failed} failed")

    # Verify shape
    sample_files = list(cache_dir.glob("*.pt"))
    if sample_files:
        t = torch.load(sample_files[0], weights_only=True)
        print(f"Stats feature shape: {t.shape} (expected: [25, 2048])")

    extractor.cleanup()


if __name__ == "__main__":
    main()
