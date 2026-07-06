"""Confidence-based trigger using Whisper avg_logprob."""

from __future__ import annotations

from accentedness_routing.triggers.base import RoutingTrigger


class ConfidenceTrigger(RoutingTrigger):
    """Escalation score = 1 - normalized_confidence.

    Uses avg_logprob from the default model: lower confidence → higher escalation score.
    Logprobs are normalized to [0, 1] using min-max over the dataset.
    """

    def __init__(self, logprobs: dict[str, float]):
        self._raw = logprobs
        vals = list(logprobs.values())
        self._min = min(vals)
        self._max = max(vals)
        rng = self._max - self._min
        if rng < 1e-8:
            rng = 1.0
        # Normalize: low logprob → high score (more reason to escalate)
        self._scores = {
            uid: 1.0 - (lp - self._min) / rng for uid, lp in logprobs.items()
        }

    @property
    def name(self) -> str:
        return "confidence"

    def score(self, utterance_id: str) -> float:
        return self._scores[utterance_id]
