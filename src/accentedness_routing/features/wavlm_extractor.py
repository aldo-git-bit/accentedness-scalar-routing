"""WavLM-large feature extraction with NaN guards."""

from __future__ import annotations

import numpy as np
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor


class WavLMExtractor:
    """Extract per-layer mean-pooled features from WavLM-large."""

    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = "cpu"):
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """Extract features from a single utterance.

        Args:
            audio: float32 waveform
            sample_rate: sample rate (must be 16000)

        Returns:
            Tensor of shape (num_layers, hidden_dim) — mean-pooled per layer.
            For WavLM-large: (25, 1024).
        """
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (1, T, D)

        # NaN guard
        for i, hs in enumerate(hidden_states):
            if not torch.isfinite(hs).all():
                raise ValueError(f"NaN/Inf detected in WavLM layer {i}")

        # Mean-pool over time dimension for each layer
        pooled = torch.stack([hs.squeeze(0).mean(dim=0) for hs in hidden_states])
        # pooled shape: (num_layers, hidden_dim)

        # Always return on CPU regardless of extraction device, so cached
        # tensors load on any machine without an explicit map_location.
        pooled = pooled.detach().cpu()

        assert torch.isfinite(pooled).all(), "NaN in pooled features"
        return pooled

    def extract_stats(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """Extract mean+std concatenated features from a single utterance.

        Args:
            audio: float32 waveform
            sample_rate: sample rate (must be 16000)

        Returns:
            Tensor of shape (num_layers, 2*hidden_dim) — [mean; std] per layer.
            For WavLM-large: (25, 2048).
        """
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (1, T, D)

        # NaN guard
        for i, hs in enumerate(hidden_states):
            if not torch.isfinite(hs).all():
                raise ValueError(f"NaN/Inf detected in WavLM layer {i}")

        # Mean and std pool over time dimension for each layer
        means = torch.stack([hs.squeeze(0).mean(dim=0) for hs in hidden_states])
        stds = torch.stack([hs.squeeze(0).std(dim=0) for hs in hidden_states])

        # Replace any NaN stds (from single-frame utterances) with 0
        stds = torch.nan_to_num(stds, nan=0.0)

        pooled = torch.cat([means, stds], dim=1)  # (num_layers, 2*hidden_dim)

        # Always return on CPU regardless of extraction device, so cached
        # tensors load on any machine without an explicit map_location.
        pooled = pooled.detach().cpu()

        assert torch.isfinite(pooled).all(), "NaN in stats-pooled features"
        return pooled

    def _feature_attention_mask(
        self, hidden_state_length: int, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Downsample a raw-sample attention mask to hidden-state time resolution.

        WavLM's conv frontend downsamples the waveform by a fixed factor
        (~320x for wavlm-large) before the transformer ever sees it, so a
        batch's raw sample-level attention_mask does not line up with
        hidden-state frames and must be re-derived at that resolution.

        Uses HF's internal _get_feature_vector_attention_mask — this is
        underscore-prefixed (not public/stable API). Verified against the
        pinned transformers==5.13.0 WavLM implementation; re-check this
        method still exists with this signature if transformers is
        upgraded, since it could move or change without a deprecation
        warning.
        """
        return self.model._get_feature_vector_attention_mask(hidden_state_length, attention_mask)

    def extract_batch(
        self, audios: list[np.ndarray], sample_rate: int = 16000
    ) -> list[torch.Tensor]:
        """Extract mean-pooled features for a batch of utterances at once.

        Pads variable-length audio to a common length and masks out
        padded frames in two places: inside the model, via attention_mask
        (so real frames don't attend to padding and get corrupted by it),
        and during pooling (so padded frames don't dilute the mean).

        Returns a list of (num_layers, hidden_dim) tensors, one per input
        utterance, in the same order as `audios`.
        """
        inputs = self.processor(
            audios, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_values, attention_mask=attention_mask, output_hidden_states=True
            )

        hidden_states = outputs.hidden_states  # tuple of (B, T, D); all layers share T

        for i, hs in enumerate(hidden_states):
            if not torch.isfinite(hs).all():
                raise ValueError(f"NaN/Inf detected in WavLM layer {i}")

        feat_mask = self._feature_attention_mask(hidden_states[0].shape[1], attention_mask)
        mask_f = feat_mask.unsqueeze(-1).to(hidden_states[0].dtype)  # (B, T, 1)
        counts = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)

        means = []
        for hs in hidden_states:
            summed = (hs * mask_f).sum(dim=1)  # (B, D)
            means.append(summed / counts)
        pooled = torch.stack(means, dim=1)  # (B, num_layers, D)

        pooled = pooled.detach().cpu()
        assert torch.isfinite(pooled).all(), "NaN in pooled features"
        return [pooled[i] for i in range(pooled.shape[0])]

    def extract_stats_batch(
        self, audios: list[np.ndarray], sample_rate: int = 16000
    ) -> list[torch.Tensor]:
        """Extract mean+std concatenated features for a batch of utterances at once.

        Same masking discipline as extract_batch. Variance is computed
        only over valid (unpadded) frames using the unbiased (N-1)
        estimator, matching torch.std's default so the single-utterance
        and batched paths agree on statistical convention and not only on
        which frames get excluded.
        """
        inputs = self.processor(
            audios, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_values, attention_mask=attention_mask, output_hidden_states=True
            )

        hidden_states = outputs.hidden_states

        for i, hs in enumerate(hidden_states):
            if not torch.isfinite(hs).all():
                raise ValueError(f"NaN/Inf detected in WavLM layer {i}")

        feat_mask = self._feature_attention_mask(hidden_states[0].shape[1], attention_mask)
        mask_f = feat_mask.unsqueeze(-1).to(hidden_states[0].dtype)  # (B, T, 1)
        counts = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
        # counts==1 => numerator (sq_diff sum) is exactly 0 too, so clamping
        # the denominator to 1 avoids a 0/0 NaN without changing the result
        # (matches extract_stats's nan_to_num(0.0) for single-frame utterances).
        unbiased_denom = (counts - 1).clamp_min(1.0)

        means, stds = [], []
        for hs in hidden_states:
            summed = (hs * mask_f).sum(dim=1)
            mean = summed / counts  # (B, D)
            sq_diff = ((hs - mean.unsqueeze(1)) ** 2) * mask_f
            var = sq_diff.sum(dim=1) / unbiased_denom
            means.append(mean)
            stds.append(var.sqrt())

        means = torch.stack(means, dim=1)  # (B, num_layers, D)
        stds = torch.stack(stds, dim=1)
        stds = torch.nan_to_num(stds, nan=0.0)

        pooled = torch.cat([means, stds], dim=2)  # (B, num_layers, 2*D)
        pooled = pooled.detach().cpu()
        assert torch.isfinite(pooled).all(), "NaN in stats-pooled features"
        return [pooled[i] for i in range(pooled.shape[0])]

    def cleanup(self):
        """Free model memory."""
        del self.model
        del self.processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()
