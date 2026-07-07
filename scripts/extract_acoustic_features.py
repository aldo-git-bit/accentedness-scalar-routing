"""Extract cheap acoustic features per utterance (no re-decode needed).

Features:
  - duration: len(audio) / sample_rate (seconds)
  - silence_ratio: fraction of frames below RMS energy threshold
  - speaking_rate: n_words_in_reference / duration (words per second)

Saves to data/acoustic_features_cache/{utterance_id}.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def compute_silence_ratio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_length_ms: int = 25,
    energy_threshold_db: float = -40.0,
) -> float:
    """Compute fraction of frames with energy below threshold.

    Uses simple RMS energy-based VAD. No external library dependency.
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)
    n_frames = len(audio) // frame_length

    if n_frames == 0:
        return 1.0

    frames = audio[:n_frames * frame_length].reshape(n_frames, frame_length)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

    # Convert threshold from dB to linear (relative to max RMS)
    max_rms = np.max(rms) if np.max(rms) > 1e-10 else 1e-10
    threshold_linear = max_rms * (10 ** (energy_threshold_db / 20))

    silent_frames = np.sum(rms < threshold_linear)
    return float(silent_frames / n_frames)


def extract_features(audio: np.ndarray, reference_text: str,
                     sample_rate: int = 16000) -> dict:
    """Extract acoustic features for a single utterance."""
    duration = len(audio) / sample_rate
    silence_ratio = compute_silence_ratio(audio, sample_rate)

    n_words = len(reference_text.split()) if reference_text else 0
    speaking_rate = n_words / duration if duration > 0 else 0.0

    return {
        "duration": duration,
        "silence_ratio": silence_ratio,
        "speaking_rate": speaking_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract acoustic features")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    cache_dir = Path(cfg.get("acoustic_features", {}).get(
        "cache_dir", "data/acoustic_features_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = cfg["data"]["sample_rate"]

    # Load all splits
    all_utterances = []
    for name in ["train", "val", "test"]:
        pkl_path = data_dir / f"{name}_utterances.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                utts = pickle.load(f)
            all_utterances.extend(utts)
            print(f"Loaded {len(utts)} {name} utterances")

    print(f"Total: {len(all_utterances)} utterances")

    n_cached = 0
    n_extracted = 0

    for utt in tqdm(all_utterances, desc="Extracting acoustic features"):
        out_path = cache_dir / f"{utt.utterance_id}.json"
        if out_path.exists():
            n_cached += 1
            continue

        features = extract_features(utt.audio, utt.text, sample_rate)
        features["utterance_id"] = utt.utterance_id

        with open(out_path, "w") as f:
            json.dump(features, f, indent=2)
        n_extracted += 1

    print(f"Done: {n_extracted} extracted, {n_cached} already cached")
    print(f"Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
