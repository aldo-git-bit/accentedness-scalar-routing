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
    """Load EdAcc, filter to target accents, resample to 16 kHz, cap per accent."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    target_accents = cfg["data"]["accents"]
    max_per_accent = cfg["data"]["utterances_per_accent"]
    target_sr = cfg["data"]["sample_rate"]

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

    # Group by accent and cap
    accent_counts: dict[str, int] = {}
    utterances: list[Utterance] = []

    for idx, sample in enumerate(full_ds):
        accent = sample["accent"]
        if accent_counts.get(accent, 0) >= max_per_accent:
            continue
        accent_counts[accent] = accent_counts.get(accent, 0) + 1

        audio_array, sr = _decode_audio(sample["audio"])
        utt = Utterance(
            utterance_id=_utterance_id(sample["speaker"], sample["text"], idx),
            speaker=sample["speaker"],
            accent=accent,
            text=sample["text"],
            audio=audio_array,
            sample_rate=sr,
        )
        utterances.append(utt)

    return utterances
