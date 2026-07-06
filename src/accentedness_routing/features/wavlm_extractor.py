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

        assert torch.isfinite(pooled).all(), "NaN in pooled features"
        return pooled

    def cleanup(self):
        """Free model memory."""
        del self.model
        del self.processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()
