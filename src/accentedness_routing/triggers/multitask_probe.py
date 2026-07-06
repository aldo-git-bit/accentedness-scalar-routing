"""Multi-task probe: shared trunk with regression + accent classification heads.

For routing, use only the regression head output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from accentedness_routing.features.pooling import LearnableWeightedSum


class MultiTaskProbe(nn.Module):
    """Multi-task probe: WavLM layers -> regression + accent classification.

    Architecture:
        LearnableWeightedSum(25) -> Linear(1024, 256) -> ReLU -> Dropout
                                    |-> Linear(256, 1) [regression head]
                                    |-> Linear(256, num_accents) [classification head]
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
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, probe_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(probe_dim, 1)
        self.classification_head = nn.Linear(probe_dim, num_accents)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_layers, hidden_dim) or (num_layers, hidden_dim)

        Returns:
            (regression_output, classification_logits)
            Regression: (batch, 1) or (1,)
            Classification: (batch, num_accents) or (num_accents,)
        """
        pooled = self.layer_pool(x)
        shared = self.trunk(pooled)
        reg = self.regression_head(shared)
        cls = self.classification_head(shared)
        return reg, cls

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """Regression head only, for routing."""
        reg, _ = self.forward(x)
        return reg
