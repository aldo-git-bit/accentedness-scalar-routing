"""Accent classifier: WavLM features -> 6-class accent prediction.

Same trunk as AccentednessProbe (uses LearnableWeightedSum) but with
a classification head instead of regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from accentedness_routing.features.pooling import LearnableWeightedSum


class AccentClassifier(nn.Module):
    """Accent classification probe: WavLM layers -> accent class.

    Architecture:
        LearnableWeightedSum(25) -> Linear(1024, 256) -> ReLU -> Dropout -> Linear(256, num_accents)
    """

    def __init__(
        self,
        num_accents: int = 6,
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
            nn.Linear(probe_dim, num_accents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_layers, hidden_dim) or (num_layers, hidden_dim)

        Returns:
            (batch, num_accents) or (num_accents,) — logits
        """
        pooled = self.layer_pool(x)
        return self.head(pooled)
