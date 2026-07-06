"""Oracle trigger: knows the true WER improvement from escalation."""

from __future__ import annotations

from accentedness_routing.triggers.base import RoutingTrigger


class OracleTrigger(RoutingTrigger):
    """Escalation score = WER_default - WER_careful (clipped to [0, 1]).

    This is the upper bound: it knows exactly which utterances benefit from escalation.
    """

    def __init__(self, default_wers: dict[str, float], careful_wers: dict[str, float]):
        self._default = default_wers
        self._careful = careful_wers

    @property
    def name(self) -> str:
        return "oracle"

    def score(self, utterance_id: str) -> float:
        gain = self._default[utterance_id] - self._careful[utterance_id]
        return max(0.0, min(1.0, gain))
