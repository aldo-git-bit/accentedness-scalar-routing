"""Load EdAcc, create speaker-disjoint splits, save manifest."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from accentedness_routing.data.edacc_loader import load_edacc
from accentedness_routing.data.splits import create_speaker_disjoint_splits, print_split_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    print("Loading EdAcc dataset...")
    utterances = load_edacc(args.config)
    print(f"Loaded {len(utterances)} utterances")

    # Print accent distribution
    from collections import Counter

    accent_counts = Counter(u.accent for u in utterances)
    for accent, count in sorted(accent_counts.items()):
        print(f"  {accent}: {count}")

    print("\nCreating speaker-disjoint splits...")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = create_speaker_disjoint_splits(
        utterances,
        output_path=str(data_dir / "splits_manifest.json"),
    )

    print_split_stats(splits)

    # Save splits as pickle for downstream use
    # (audio arrays are large, so we save the full utterance objects)
    for split_name, split_utts in splits.items():
        out_path = data_dir / f"{split_name}_utterances.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(split_utts, f)
        print(f"Saved {split_name} -> {out_path} ({len(split_utts)} utterances)")

    # Also save a lightweight index (no audio) as JSON
    index = {}
    for split_name, split_utts in splits.items():
        index[split_name] = [
            {
                "utterance_id": u.utterance_id,
                "speaker": u.speaker,
                "accent": u.accent,
                "text": u.text,
                "sample_rate": u.sample_rate,
                "audio_len_samples": len(u.audio),
            }
            for u in split_utts
        ]
    with open(data_dir / "utterance_index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved utterance index -> {data_dir / 'utterance_index.json'}")


if __name__ == "__main__":
    main()
