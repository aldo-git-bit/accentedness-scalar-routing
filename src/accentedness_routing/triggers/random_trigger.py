"""Random trigger baseline."""

from __future__ import annotations

import random

from accentedness_routing.triggers.base import RoutingTrigger


class RandomTrigger(RoutingTrigger):
    """Uniform random score in [0, 1]. Represents no-information baseline."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._scores: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "random"

    def score(self, utterance_id: str) -> float:
        if utterance_id not in self._scores:
            self._scores[utterance_id] = self._rng.random()
        return self._scores[utterance_id]
