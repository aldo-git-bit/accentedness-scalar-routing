"""Decode utterances with the Indian-accent finetuned Whisper model.

Uses PyTorch transformers pipeline (not MLX) since the model is a PyTorch
finetune. Runs on MPS with PYTORCH_ENABLE_MPS_FALLBACK=1.

Model: Tejveer12/Indian-Accent-English-Whisper-Finetuned
Base: whisper-large-v3-turbo finetuned on Indian-accented English

Caches results under data/asr_cache/Tejveer12_Indian_Accent_English_Whisper_Finetuned/

Gate D: Confirm before running (downloads ~3GB model).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from accentedness_routing.asr.cache import load_cached, save_cached
from accentedness_routing.asr.wer import compute_wer


def run_adapted_model(utterances, model_id: str, cache_dir: str):
    """Transcribe utterances with PyTorch transformers pipeline."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "mps" else torch.float32

    print(f"Loading model {model_id} on {device}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch_dtype,
    )

    n_cached = 0
    n_transcribed = 0

    for utt in tqdm(utterances, desc=f"ASR [{model_id.split('/')[-1]}]"):
        cached = load_cached(cache_dir, model_id, utt.utterance_id)
        if cached is not None:
            n_cached += 1
            continue

        audio = utt.audio.astype(np.float32)
        start = time.monotonic()

        result = pipe(
            audio,
            generate_kwargs={"language": "en", "task": "transcribe"},
            return_timestamps=False,
        )
        wall_clock = time.monotonic() - start

        text = result.get("text", "").strip()
        wer = compute_wer(utt.text, text)

        asr_result = {
            "text": text,
            "wer": wer,
            "reference": utt.text,
            "utterance_id": utt.utterance_id,
            "accent": utt.accent,
            "speaker": utt.speaker,
            "wall_clock_seconds": wall_clock,
            "avg_logprob": None,  # Not available from pipeline
            "no_speech_prob": None,
        }

        save_cached(cache_dir, model_id, utt.utterance_id, asr_result)
        n_transcribed += 1

    print(f"  {model_id}: {n_transcribed} transcribed, {n_cached} cached")


def main():
    parser = argparse.ArgumentParser(description="Decode with Indian-accent adapted model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["asr"]["cache_dir"]
    model_id = "Tejveer12/Indian-Accent-English-Whisper-Finetuned"

    data_dir = Path("data")
    splits_to_run = [args.split] if args.split else ["test"]

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
    run_adapted_model(all_utterances, model_id, cache_dir)


if __name__ == "__main__":
    main()
