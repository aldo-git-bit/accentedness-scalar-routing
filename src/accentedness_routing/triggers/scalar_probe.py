"""Accentedness/difficulty probe: WavLM features → scalar WER prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from accentedness_routing.features.pooling import LearnableWeightedSum
from accentedness_routing.triggers.base import RoutingTrigger


class AccentednessProbe(nn.Module):
    """Linear probe: WavLM layers → scalar difficulty score.

    Architecture:
        LearnableWeightedSum(25) → Linear(1024, 256) → ReLU → Dropout → Linear(256, 1)
    """

    def __init__(
        self,
        num_layers: int = 25,
        hidden_dim: int = 1024,
        probe_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_pool = LearnableWeightedSum(num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, probe_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(probe_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_layers, hidden_dim) or (num_layers, hidden_dim)

        Returns:
            (batch, 1) or (1,) — predicted WER
        """
        pooled = self.layer_pool(x)
        return self.head(pooled)


class ScalarProbeTrigger(RoutingTrigger):
    """Routing trigger backed by a trained AccentednessProbe."""

    def __init__(
        self,
        model: AccentednessProbe,
        features: dict[str, torch.Tensor],
        calibration: dict | None = None,
    ):
        """
        Args:
            model: trained probe
            features: utterance_id → (num_layers, hidden_dim) tensor
            calibration: dict with 'low' and 'high' percentile values for normalization
        """
        self._model = model
        self._model.eval()
        self._features = features
        self._cal = calibration or {"low": 0.0, "high": 1.0}

        # Pre-compute all scores
        self._scores: dict[str, float] = {}
        with torch.no_grad():
            for uid, feat in features.items():
                raw = self._model(feat.unsqueeze(0)).item()
                self._scores[uid] = self._calibrate(raw)

    def _calibrate(self, raw: float) -> float:
        """Normalize raw prediction to [0, 1] using percentile calibration."""
        low, high = self._cal["low"], self._cal["high"]
        rng = high - low
        if rng < 1e-8:
            return 0.5
        normed = (raw - low) / rng
        return max(0.0, min(1.0, normed))

    @property
    def name(self) -> str:
        return "scalar_probe"

    def score(self, utterance_id: str) -> float:
        return self._scores[utterance_id]
