"""EXP-13: WavLM-large (trainable) + the existing AccentednessProbe head.

Composes three pieces, none of them modified: WavLMModel (unfrozen or
frozen, per flags), the grad-enabled masked pooling in
features/masked_pooling.py, and the existing, untouched AccentednessProbe
(scalar_probe.py) as a submodule. The only new code here is the composition
and the freeze/mode bookkeeping needed to make the frozen-encoder control
faithfully reproduce the original frozen-feature pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import WavLMConfig, WavLMModel

from accentedness_routing.features.masked_pooling import masked_mean_pool
from accentedness_routing.triggers.base import RoutingTrigger
from accentedness_routing.triggers.scalar_probe import AccentednessProbe


class WavLMGainProbe(nn.Module):
    """WavLM-large -> masked mean-pool (gradient-enabled) -> AccentednessProbe.

    freeze_encoder=True freezes every WavLM parameter (feature extractor,
    projection, and transformer) — this is the frozen-encoder control,
    which must reproduce probe_gain.pt's numbers at world_size=1 with the
    same seed and effective batch size.

    freeze_feature_extractor=True (default, only relevant when
    freeze_encoder=False) freezes only the conv frontend
    (self.wavlm.feature_extractor) — standard wav2vec2/WavLM fine-tuning
    practice; the projection layer and transformer stack remain trainable.

    disable_train_augmentation=True (default) turns off WavLM's own
    train-mode-only internal behavior: SpecAugment time-masking
    (config.apply_spec_augment, mask_time_prob=0.075 by default — this
    also fixes a real crash on short utterances, since SpecAugment's
    mask_time_length=10 requires a post-downsampling sequence length
    greater than 10 frames, which the shortest utterances in this dataset
    don't have) and LayerDrop (config.layerdrop=0.1 by default —
    stochastically skips whole transformer layers during training). Both
    are audited (grep for every `self.training`-conditional branch in
    modeling_wavlm.py) to be the *only* two behavioral differences between
    train and eval mode beyond ordinary dropout; two other train-mode
    branches exist in the source (WavLMGumbelVectorQuantizer,
    WavLMAdapter's layer-skip) but neither is reachable from WavLMModel's
    forward path in this configuration (add_adapter=False; the quantizer
    is only used by WavLMForPreTraining, which isn't the class we load).

    This is a comparison-cleanliness choice, not just a crash fix: the
    frozen-encoder control (train_probe_ext1.py) trains with none of this
    — no SpecAugment, no LayerDrop, since it never runs anything through
    WavLM in train() mode at all. Leaving either active for the unfrozen
    run would introduce an unaudited confound into the frozen-vs-unfrozen
    comparison, on top of whatever the encoder-unfreezing itself changes.
    Re-enabling SpecAugment (sized to this dataset's shortest utterance,
    not the default mask_time_length=10) is a legitimate follow-up variant
    — but only if the train/val gap on the A40 run shows overfitting, not
    something to reach for preemptively.
    """

    def __init__(
        self,
        model_name: str,
        num_layers: int,
        hidden_dim: int,
        probe_dim: int,
        dropout: float,
        freeze_feature_extractor: bool = True,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        disable_train_augmentation: bool = True,
    ):
        super().__init__()
        # self.probe is constructed FIRST, deliberately — WavLMModel.from_pretrained
        # randomly initializes ~300M parameters before overwriting them with the
        # pretrained checkpoint (standard PyTorch pattern), consuming a large
        # number of RNG draws in the process. Building self.wavlm first would
        # shift self.probe's random init onto a different point in the RNG
        # stream than a standalone AccentednessProbe(...) constructed right
        # after the same torch.manual_seed(seed) call — silently breaking the
        # frozen-encoder control's ability to reproduce probe_gain.pt, "same
        # seed" notwithstanding. Confirmed empirically: constructing the two
        # orderings produced different initial probe weights before any
        # training happened.
        self.probe = AccentednessProbe(num_layers, hidden_dim, probe_dim, dropout)
        wavlm_config = WavLMConfig.from_pretrained(model_name)
        if disable_train_augmentation:
            wavlm_config.apply_spec_augment = False
            wavlm_config.layerdrop = 0.0
        self.wavlm = WavLMModel.from_pretrained(model_name, config=wavlm_config)
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.wavlm.parameters():
                p.requires_grad = False
        elif freeze_feature_extractor:
            for p in self.wavlm.feature_extractor.parameters():
                p.requires_grad = False

        if gradient_checkpointing and not freeze_encoder:
            self.wavlm.gradient_checkpointing_enable()

    def train(self, mode: bool = True):
        """Override: self.wavlm must stay in eval() when frozen, regardless
        of the outer mode, to faithfully reproduce the original
        frozen-feature pipeline (WavLMExtractor always ran with
        model.eval(), never trained it — deterministic, no dropout). Without
        this override, a blanket .train() call would put a "frozen" WavLM
        into train mode (enabling its internal dropout) even though its
        weights never update, which would NOT reproduce probe_gain.pt's
        frozen-feature numbers even at world_size=1 with identical seeding.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.wavlm.eval()
        return self

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Deliberately no torch.no_grad() here — freezing is handled entirely
        # via requires_grad=False above, which lets autograd naturally skip
        # building a graph for the frozen parts while still tracking the
        # trainable ones (the probe head always, the encoder when unfrozen).
        # An explicit no_grad wrapper here would silently zero ALL gradients,
        # including the probe head's — the single most dangerous failure
        # mode carried over from WavLMExtractor's (correct, for its use)
        # no_grad-wrapped extraction methods.
        outputs = self.wavlm(input_values, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, T, D)
        feat_mask = self.wavlm._get_feature_vector_attention_mask(hidden_states[0].shape[1], attention_mask)
        pooled = masked_mean_pool(hidden_states, feat_mask)  # (B, num_layers, D)
        return self.probe(pooled)  # (B, 1)


class DDPGainTrigger(RoutingTrigger):
    """RoutingTrigger wrapper for a trained WavLMGainProbe.

    Mirrors ScalarProbeTrigger's API and calibration convention exactly, so
    it plugs into eval_common.py's operating_curve/summarize/bootstrap
    unchanged — but runs inference on raw audio through the (trainable)
    encoder rather than reading precomputed cached features.
    """

    def __init__(
        self,
        model: WavLMGainProbe,
        utterances: list,
        processor,
        calibration: dict | None,
        device: torch.device,
    ):
        self._model = model
        self._model.eval()
        self._cal = calibration or {"low": 0.0, "high": 1.0}

        self._scores: dict[str, float] = {}
        with torch.no_grad():
            for utt in utterances:
                inputs = processor(utt.audio, sampling_rate=utt.sample_rate, return_tensors="pt")
                input_values = inputs.input_values.to(device)
                attention_mask = torch.ones_like(input_values, dtype=torch.long)
                raw = self._model(input_values, attention_mask).item()
                self._scores[utt.utterance_id] = self._calibrate(raw)

    def _calibrate(self, raw: float) -> float:
        low, high = self._cal["low"], self._cal["high"]
        rng = high - low
        if rng < 1e-8:
            return 0.5
        normed = (raw - low) / rng
        return max(0.0, min(1.0, normed))

    @property
    def name(self) -> str:
        return "wavlm_unfrozen_gain"

    def score(self, utterance_id: str) -> float:
        return self._scores[utterance_id]
