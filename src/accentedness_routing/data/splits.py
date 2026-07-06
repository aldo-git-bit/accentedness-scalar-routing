"""Speaker-disjoint train/val/test splits."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from accentedness_routing.data.edacc_loader import Utterance


def create_speaker_disjoint_splits(
    utterances: list[Utterance],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    output_path: str | None = None,
) -> dict[str, list[Utterance]]:
    """Split utterances into train/val/test with no speaker overlap.

    All utterances from a given speaker go to exactly one fold.
    Speakers are allocated per-accent to maintain accent balance.
    """
    # Group speakers by accent
    accent_speakers: dict[str, set[str]] = defaultdict(set)
    speaker_utts: dict[str, list[Utterance]] = defaultdict(list)

    for utt in utterances:
        accent_speakers[utt.accent].add(utt.speaker)
        speaker_utts[utt.speaker].append(utt)

    splits: dict[str, list[Utterance]] = {"train": [], "val": [], "test": []}
    manifest: dict[str, dict] = {}

    rng = random.Random(seed)

    for accent, speakers in sorted(accent_speakers.items()):
        speaker_list = sorted(speakers)
        rng.shuffle(speaker_list)

        n = len(speaker_list)
        if n < 3:
            raise ValueError(
                f"Accent '{accent}' has only {n} speaker(s); need at least 3 "
                f"for speaker-disjoint train/val/test splits."
            )

        # Guarantee at least 1 speaker per fold, then distribute remainder
        # by ratio. Reserve 1 each for val and test first, rest to train.
        n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test
        # Safety: if rounding left train with 0, take from the largest group
        if n_train < 1:
            n_train = 1
            if n_val > n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        train_spk = speaker_list[:n_train]
        val_spk = speaker_list[n_train : n_train + n_val]
        test_spk = speaker_list[n_train + n_val :]

        for spk in train_spk:
            splits["train"].extend(speaker_utts[spk])
        for spk in val_spk:
            splits["val"].extend(speaker_utts[spk])
        for spk in test_spk:
            splits["test"].extend(speaker_utts[spk])

        manifest[accent] = {
            "train_speakers": train_spk,
            "val_speakers": val_spk,
            "test_speakers": test_spk,
        }

    # Assert zero speaker overlap
    train_spk_set = {u.speaker for u in splits["train"]}
    val_spk_set = {u.speaker for u in splits["val"]}
    test_spk_set = {u.speaker for u in splits["test"]}
    assert train_spk_set.isdisjoint(val_spk_set), "Speaker overlap: train ∩ val"
    assert train_spk_set.isdisjoint(test_spk_set), "Speaker overlap: train ∩ test"
    assert val_spk_set.isdisjoint(test_spk_set), "Speaker overlap: val ∩ test"

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Build serializable manifest
        full_manifest = {
            "seed": seed,
            "splits": {},
            "speaker_assignment": manifest,
        }
        for split_name, split_utts in splits.items():
            full_manifest["splits"][split_name] = {
                "n_utterances": len(split_utts),
                "n_speakers": len({u.speaker for u in split_utts}),
                "utterance_ids": [u.utterance_id for u in split_utts],
            }
        with open(output_path, "w") as f:
            json.dump(full_manifest, f, indent=2)

    return splits


def print_split_stats(splits: dict[str, list[Utterance]]) -> None:
    """Print summary statistics for splits."""
    for name, utts in splits.items():
        speakers = {u.speaker for u in utts}
        accents = defaultdict(int)
        for u in utts:
            accents[u.accent] += 1
        print(f"\n{name}: {len(utts)} utterances, {len(speakers)} speakers")
        for accent, count in sorted(accents.items()):
            print(f"  {accent}: {count}")
