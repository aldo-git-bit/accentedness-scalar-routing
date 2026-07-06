"""Hallucination-based routing triggers.

Two triggers using only cached ASR data (no re-extraction):
  - CompressionRatioTrigger: zlib compression ratio of hypothesis text
  - NoSpeechProbTrigger: no_speech_prob field from Whisper
"""

from __future__ import annotations

import zlib

from accentedness_routing.triggers.base import RoutingTrigger


class CompressionRatioTrigger(RoutingTrigger):
    """Compression ratio of hypothesis text as hallucination proxy.

    Hallucinated transcripts tend to be highly repetitive, yielding
    lower compression ratios (compressed/raw is smaller for non-repetitive text,
    but higher raw/compressed for repetitive text).

    Score = normalized raw/compressed ratio. Higher = more repetitive = escalate.
    """

    def __init__(self, hypotheses: dict[str, str]):
        """
        Args:
            hypotheses: utterance_id -> hypothesis text from default model.
        """
        ratios: dict[str, float] = {}
        for uid, text in hypotheses.items():
            encoded = text.encode("utf-8")
            if len(encoded) == 0:
                ratios[uid] = 1.0  # empty text is suspicious
            else:
                compressed = zlib.compress(encoded)
                ratios[uid] = len(encoded) / len(compressed)

        # Normalize to [0, 1]
        vals = list(ratios.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx - mn > 1e-8 else 1.0
        self._scores = {uid: (r - mn) / rng for uid, r in ratios.items()}

    @property
    def name(self) -> str:
        return "compression_ratio"

    def score(self, utterance_id: str) -> float:
        return self._scores[utterance_id]


class NoSpeechProbTrigger(RoutingTrigger):
    """No-speech probability as hallucination proxy.

    Whisper's no_speech_prob indicates likelihood that the segment
    contains no speech. High values suggest the model may be
    hallucinating content from silence.

    Score = normalized no_speech_prob. Higher = escalate.
    """

    def __init__(self, no_speech_probs: dict[str, float]):
        """
        Args:
            no_speech_probs: utterance_id -> no_speech_prob from default model.
        """
        vals = list(no_speech_probs.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx - mn > 1e-8 else 1.0
        self._scores = {uid: (p - mn) / rng for uid, p in no_speech_probs.items()}

    @property
    def name(self) -> str:
        return "no_speech_prob"

    def score(self, utterance_id: str) -> float:
        return self._scores[utterance_id]
