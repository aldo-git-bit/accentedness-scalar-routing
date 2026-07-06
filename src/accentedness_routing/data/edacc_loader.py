"""Load and filter the EdAcc dataset from HuggingFace."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import yaml
from datasets import Audio, concatenate_datasets, load_dataset


@dataclass
class Utterance:
    utterance_id: str
    speaker: str
    accent: str
    text: str
    audio: np.ndarray  # float32, 16 kHz
    sample_rate: int


# Mapping from config accent names to EdAcc accent field values.
ACCENT_MAP = {
    "american": "Mainstream US English",
    "southern_english": "Southern British English",
    "irish": "Irish English",
    "scottish": "Scottish English",
    "indian": "Indian English",
    "nigerian": "Nigerian English",
}


def _utterance_id(speaker: str, text: str, idx: int) -> str:
    """Deterministic ID from speaker + text + index."""
    raw = f"{speaker}|{text}|{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _decode_audio(audio_obj) -> tuple[np.ndarray, int]:
    """Extract numpy array and sample rate from a datasets audio object.

    Handles both the legacy dict format and the newer torchcodec AudioDecoder.
    """
    if isinstance(audio_obj, dict):
        return np.array(audio_obj["array"], dtype=np.float32), audio_obj["sampling_rate"]

    # torchcodec AudioDecoder (datasets >= 3.x with torchcodec)
    samples = audio_obj.get_all_samples()
    arr = samples.data.squeeze(0).numpy().astype(np.float32)
    return arr, samples.sample_rate


def load_edacc(config_path: str) -> list[Utterance]:
    """Load EdAcc, filter to target accents, resample to 16 kHz.

    Samples utterances evenly across speakers within each accent to ensure
    all speakers are represented (critical for speaker-disjoint splits).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    target_accents = cfg["data"]["accents"]
    max_per_accent = cfg["data"]["utterances_per_accent"]
    target_sr = cfg["data"]["sample_rate"]
    seed = cfg.get("seed", 42)

    # Map config names to dataset values
    accent_values = {ACCENT_MAP[a] for a in target_accents}

    # Load both splits (EdAcc has no train split)
    val_ds = load_dataset("edinburghcstr/edacc", split="validation")
    test_ds = load_dataset("edinburghcstr/edacc", split="test")
    full_ds = concatenate_datasets([val_ds, test_ds])

    # Cast audio to target sample rate
    full_ds = full_ds.cast_column("audio", Audio(sampling_rate=target_sr))

    # Filter to target accents
    full_ds = full_ds.filter(lambda x: x["accent"] in accent_values)

    # First pass: collect indices grouped by (accent, speaker)
    from collections import defaultdict
    accent_speaker_indices: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx in range(len(full_ds)):
        sample = full_ds[idx]
        accent_speaker_indices[sample["accent"]][sample["speaker"]].append(idx)

    # Second pass: sample evenly across speakers within each accent
    import random
    rng = random.Random(seed)
    selected_indices: list[int] = []

    for accent_val in sorted(accent_speaker_indices):
        speakers = accent_speaker_indices[accent_val]
        speaker_list = sorted(speakers.keys())
        n_speakers = len(speaker_list)

        # Shuffle each speaker's utterances, then round-robin sample
        per_speaker_indices = {}
        for spk in speaker_list:
            idxs = list(speakers[spk])
            rng.shuffle(idxs)
            per_speaker_indices[spk] = idxs

        # Round-robin across speakers until we hit the cap
        accent_selected = []
        pointer = {spk: 0 for spk in speaker_list}
        while len(accent_selected) < max_per_accent:
            added_this_round = False
            for spk in speaker_list:
                if pointer[spk] < len(per_speaker_indices[spk]):
                    accent_selected.append(per_speaker_indices[spk][pointer[spk]])
                    pointer[spk] += 1
                    added_this_round = True
                    if len(accent_selected) >= max_per_accent:
                        break
            if not added_this_round:
                break  # All speakers exhausted

        selected_indices.extend(accent_selected)

    # Load audio for selected indices
    utterances: list[Utterance] = []
    for idx in selected_indices:
        sample = full_ds[idx]
        audio_array, sr = _decode_audio(sample["audio"])
        utt = Utterance(
            utterance_id=_utterance_id(sample["speaker"], sample["text"], idx),
            speaker=sample["speaker"],
            accent=sample["accent"],
            text=sample["text"],
            audio=audio_array,
            sample_rate=sr,
        )
        utterances.append(utt)

    return utterances
