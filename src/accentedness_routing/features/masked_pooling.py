"""Gradient-enabled masked mean-pooling, for use inside a training forward pass.

This is the same masked-mean math verified in WavLMExtractor.extract_batch
(Stage 2: padding-correlation signature eliminated, r~0.02-0.06, both
statistically indistinguishable from zero) — but WavLMExtractor's methods
wrap the model call in torch.no_grad() and .detach().cpu() the output,
which is correct for frozen-feature caching and *wrong* for a trainable
forward pass: no_grad silently zeroes every gradient (the model still
"trains" without error, it just never updates), and detach/cpu would sever
the graph and move off-device. WavLMExtractor is intentionally left
untouched (Stage 2, verified); this module holds only the reusable math,
with no no_grad/detach/cpu anywhere.
"""

from __future__ import annotations

import torch


def masked_mean_pool(hidden_states: tuple[torch.Tensor, ...], feat_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool each layer's hidden states over valid (unpadded) frames only.

    Args:
        hidden_states: tuple of (B, T, D) tensors, one per WavLM layer
            (output_hidden_states=True), all sharing the same T.
        feat_mask: (B, T) bool tensor at hidden-state time resolution (see
            WavLMModel._get_feature_vector_attention_mask — the raw
            sample-level attention_mask does not line up with T after the
            conv frontend's ~320x downsampling, so this must already be at
            the right resolution; computing it is the caller's job, since
            it requires the model instance).

    Returns:
        (B, num_layers, D) — gradients flow through if hidden_states does.
    """
    mask_f = feat_mask.unsqueeze(-1).to(hidden_states[0].dtype)  # (B, T, 1)
    counts = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)

    means = []
    for hs in hidden_states:
        summed = (hs * mask_f).sum(dim=1)  # (B, D)
        means.append(summed / counts)
    return torch.stack(means, dim=1)  # (B, num_layers, D)
