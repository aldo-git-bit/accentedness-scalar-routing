"""Transcribe all utterances with both models and cache results."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import yaml
from tqdm import tqdm

from accentedness_routing.asr.cache import load_cached, save_cached
from accentedness_routing.asr.transcribe import transcribe_utterance
from accentedness_routing.asr.wer import compute_wer


def run_model(utterances, model_path: str, language: str, cache_dir: str):
    """Transcribe all utterances with one model, using cache."""
    n_cached = 0
    n_transcribed = 0

    for utt in tqdm(utterances, desc=f"ASR [{model_path.split('/')[-1]}]"):
        cached = load_cached(cache_dir, model_path, utt.utterance_id)
        if cached is not None:
            n_cached += 1
            continue

        result = transcribe_utterance(utt.audio, model_path, language)
        result["wer"] = compute_wer(utt.text, result["text"])
        result["reference"] = utt.text
        result["utterance_id"] = utt.utterance_id
        result["accent"] = utt.accent
        result["speaker"] = utt.speaker

        save_cached(cache_dir, model_path, utt.utterance_id, result)
        n_transcribed += 1

    print(f"  {model_path}: {n_transcribed} transcribed, {n_cached} cached")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default=None, help="Run only on this split (train/val/test)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["asr"]["cache_dir"]
    language = cfg["asr"]["language"]
    models = [cfg["asr"]["default_model"], cfg["asr"]["careful_model"]]

    # Load utterances from all splits
    data_dir = Path("data")
    splits_to_run = [args.split] if args.split else ["train", "val", "test"]

    all_utterances = []
    for split_name in splits_to_run:
        pkl_path = data_dir / f"{split_name}_utterances.pkl"
        if not pkl_path.exists():
            print(f"Warning: {pkl_path} not found, skipping")
            continue
        with open(pkl_path, "rb") as f:
            utts = pickle.load(f)
        all_utterances.extend(utts)
        print(f"Loaded {len(utts)} utterances from {split_name}")

    print(f"\nTotal utterances: {len(all_utterances)}")

    # Run each model sequentially (never co-load both large models)
    for model_path in models:
        print(f"\nProcessing with {model_path}...")
        run_model(all_utterances, model_path, language, cache_dir)


if __name__ == "__main__":
    main()
