"""Disk cache for ASR results, keyed by utterance_id + model_id."""

from __future__ import annotations

import json
from pathlib import Path


def _model_slug(model_path: str) -> str:
    """Convert model path to filesystem-safe slug."""
    return model_path.replace("/", "_").replace("-", "_")


def cache_path(cache_dir: str, model_path: str, utterance_id: str) -> Path:
    """Return path to cached result file."""
    slug = _model_slug(model_path)
    return Path(cache_dir) / slug / f"{utterance_id}.json"


def load_cached(cache_dir: str, model_path: str, utterance_id: str) -> dict | None:
    """Load cached ASR result, or None if not cached."""
    path = cache_path(cache_dir, model_path, utterance_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_cached(cache_dir: str, model_path: str, utterance_id: str, result: dict) -> None:
    """Save ASR result to cache."""
    path = cache_path(cache_dir, model_path, utterance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
