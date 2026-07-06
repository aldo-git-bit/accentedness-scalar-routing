"""Extract WavLM-large features for all utterances and cache to disk."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from accentedness_routing.features.wavlm_extractor import WavLMExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["features"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["features"]["model"]
    device = cfg["features"]["device"]

    # Load all utterances
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
    print(f"Loading {model_name} on {device}...")
    extractor = WavLMExtractor(model_name, device)

    n_cached = 0
    n_extracted = 0
    n_failed = 0

    for utt in tqdm(all_utterances, desc="Extracting features"):
        out_path = cache_dir / f"{utt.utterance_id}.pt"
        if out_path.exists():
            n_cached += 1
            continue

        try:
            features = extractor.extract(utt.audio, utt.sample_rate)
            torch.save(features, out_path)
            n_extracted += 1
        except ValueError as e:
            print(f"\n  SKIP {utt.utterance_id}: {e}")
            n_failed += 1

    print(f"\nDone: {n_extracted} extracted, {n_cached} cached, {n_failed} failed")

    # Verify shape of a random cached file
    sample_files = list(cache_dir.glob("*.pt"))
    if sample_files:
        t = torch.load(sample_files[0], weights_only=True)
        print(f"Feature shape: {t.shape} (expected: [{cfg['features']['num_layers']}, {cfg['features']['hidden_dim']}])")

    extractor.cleanup()


if __name__ == "__main__":
    main()
