"""Abstract base class for routing triggers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RoutingTrigger(ABC):
    """Produces a scalar score in [0, 1] per utterance.

    Higher score = more reason to escalate to the careful model.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def score(self, utterance_id: str) -> float:
        """Return routing score for a single utterance."""
        ...

    def score_batch(self, utterance_ids: list[str]) -> list[float]:
        """Score a batch of utterances. Default: iterate."""
        return [self.score(uid) for uid in utterance_ids]
