"""Learnable weighted sum over WavLM layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnableWeightedSum(nn.Module):
    """Learnable weighted sum over a stack of layer representations.

    Given input of shape (num_layers, hidden_dim), produces (hidden_dim,)
    using softmax-normalized learnable weights.
    """

    def __init__(self, num_layers: int = 25):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_layers, hidden_dim) or (num_layers, hidden_dim)

        Returns:
            (batch, hidden_dim) or (hidden_dim,)
        """
        normed = torch.softmax(self.weights, dim=0)

        if x.dim() == 3:
            # (batch, num_layers, hidden_dim)
            return (x * normed[None, :, None]).sum(dim=1)
        else:
            # (num_layers, hidden_dim)
            return (x * normed[:, None]).sum(dim=0)

    def get_layer_weights(self) -> list[float]:
        """Return normalized layer weights as a list."""
        with torch.no_grad():
            return torch.softmax(self.weights, dim=0).tolist()
