# Research Objective

Build and evaluate an accentedness-scalar router for English ASR that decides per utterance whether to use a fast default recognizer (Whisper-small) or escalate to a careful one (Whisper-large-v3).

The scalar is learned from WavLM-large features via a linear probe that regresses base-model per-utterance WER.

**Success criterion:** The scalar's operating curve (net WER vs escalation rate) dominates the argmax-classifier baseline.
