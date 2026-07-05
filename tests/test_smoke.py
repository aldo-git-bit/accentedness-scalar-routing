"""Smoke tests: verify core dependencies load and produce finite outputs."""

import numpy as np


def test_wavlm_loads():
    """WavLM-large loads and produces finite hidden states."""
    import torch
    from transformers import WavLMModel

    model = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cpu")
    model.eval()
    dummy = torch.randn(1, 16000)
    with torch.no_grad():
        out = model(dummy, output_hidden_states=True)
    for i, hs in enumerate(out.hidden_states):
        assert torch.isfinite(hs).all(), f"NaN in WavLM layer {i}"
    print(f"WavLM OK: {len(out.hidden_states)} layers, shape {out.hidden_states[0].shape}")


def test_mlx_whisper_transcribes():
    """mlx-whisper transcribes a short audio clip."""
    import mlx_whisper

    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo="mlx-community/whisper-small-mlx", language="en"
    )
    assert "text" in result
    print(f"mlx-whisper OK: transcribed to '{result['text'][:50]}'")


def test_jiwer_wer():
    """jiwer computes WER correctly."""
    import jiwer

    transforms = jiwer.Compose(
        [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip(), jiwer.RemoveMultipleSpaces()]
    )
    ref = "the cat sat on the mat"
    hyp = "the cat on the mat"
    wer = jiwer.wer(ref, hyp, truth_transform=transforms, hypothesis_transform=transforms)
    assert 0 < wer < 1
    print(f"jiwer OK: WER={wer:.3f}")


if __name__ == "__main__":
    test_jiwer_wer()
    print("Smoke test: jiwer passed")
    test_mlx_whisper_transcribes()
    print("Smoke test: mlx-whisper passed")
    test_wavlm_loads()
    print("Smoke test: WavLM passed")
    print("All smoke tests passed!")
