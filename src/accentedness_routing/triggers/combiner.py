"""Composite combiner trigger using a fitted sklearn model.

Combines multiple routing signals (confidence, no_speech_prob, champion score,
acoustic features) into a single routing score via logistic regression.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from accentedness_routing.triggers.base import RoutingTrigger


class CombinerTrigger(RoutingTrigger):
    """Routing trigger backed by a fitted sklearn classifier.

    Takes a fitted model (e.g., LogisticRegression) and a feature-extraction
    function. For each utterance, extracts features and returns
    model.predict_proba(features)[1] as the routing score.
    """

    def __init__(
        self,
        model: Any,
        feature_fn: Callable[[str], np.ndarray],
        trigger_name: str = "combiner",
    ):
        """
        Args:
            model: Fitted sklearn model with predict_proba method.
            feature_fn: Callable that takes utterance_id and returns
                a 1-D feature array.
            trigger_name: Name for this trigger instance.
        """
        self._model = model
        self._feature_fn = feature_fn
        self._name = trigger_name

    @property
    def name(self) -> str:
        return self._name

    def score(self, utterance_id: str) -> float:
        features = self._feature_fn(utterance_id)
        proba = self._model.predict_proba(features.reshape(1, -1))
        return float(proba[0, 1])

    def score_batch(self, utterance_ids: list[str]) -> list[float]:
        """Score a batch efficiently via matrix prediction."""
        features = np.array([self._feature_fn(uid) for uid in utterance_ids])
        proba = self._model.predict_proba(features)
        return proba[:, 1].tolist()
