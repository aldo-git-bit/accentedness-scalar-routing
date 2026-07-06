"""WER computation with text normalization."""

from __future__ import annotations

import jiwer


def _make_transforms() -> jiwer.Compose:
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(),
    ])


_TRANSFORMS = _make_transforms()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute WER between reference and hypothesis after normalization.

    Returns WER as a float (0.0 = perfect, can be > 1.0 for very bad hypotheses).
    """
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0

    return float(jiwer.wer(
        reference,
        hypothesis,
        reference_transform=_TRANSFORMS,
        hypothesis_transform=_TRANSFORMS,
    ))
