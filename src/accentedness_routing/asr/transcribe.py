"""mlx-whisper transcription wrapper."""

from __future__ import annotations

import time

import mlx_whisper
import numpy as np


def transcribe_utterance(
    audio: np.ndarray,
    model_path: str,
    language: str = "en",
) -> dict:
    """Transcribe a single utterance with mlx-whisper.

    Returns dict with: text, segments, avg_logprob, no_speech_prob, wall_clock_seconds.
    """
    audio = audio.astype(np.float32)

    start = time.monotonic()
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model_path,
        language=language,
    )
    wall_clock = time.monotonic() - start

    # Extract segment-level info
    segments = result.get("segments", [])
    avg_logprobs = [s.get("avg_logprob", 0.0) for s in segments]
    no_speech_probs = [s.get("no_speech_prob", 0.0) for s in segments]

    return {
        "text": result.get("text", "").strip(),
        "avg_logprob": float(np.mean(avg_logprobs)) if avg_logprobs else 0.0,
        "no_speech_prob": float(np.mean(no_speech_probs)) if no_speech_probs else 0.0,
        "wall_clock_seconds": wall_clock,
        "n_segments": len(segments),
    }
