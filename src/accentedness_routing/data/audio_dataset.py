"""Raw-audio Dataset for EXP-13 training.

Every existing training script in this repo reads pre-extracted, cached
feature tensors (data/features_cache/*.pt) — that only works for a frozen
encoder. EXP-13 fine-tunes WavLM itself, so features can't be cached;
training needs raw audio, re-run through the encoder every step.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class AudioGainDataset(Dataset):
    """Wraps a list of Utterance objects + a utterance_id -> target dict."""

    def __init__(self, utterances: list, targets: dict[str, float]):
        self.utterances = utterances
        self.targets = targets

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        utt = self.utterances[idx]
        return {
            "audio": utt.audio,
            "sample_rate": utt.sample_rate,
            "target": self.targets[utt.utterance_id],
            "utterance_id": utt.utterance_id,
        }


class Collator:
    """Pads a batch of variable-length audio via the same HF processor and
    padding=True convention WavLMExtractor.extract_batch uses (Stage 2,
    verified), so batches built here are masked identically to the verified
    extraction path.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        audios = [item["audio"] for item in batch]
        sample_rate = batch[0]["sample_rate"]
        inputs = self.processor(audios, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        targets = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
        utterance_ids = [item["utterance_id"] for item in batch]
        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "targets": targets,
            "utterance_ids": utterance_ids,
        }
