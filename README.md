# Accentedness-Scalar Routing for English ASR

Route English ASR utterances between a fast default recognizer (Whisper-small) and a careful one (Whisper-large-v3) using a learned difficulty scalar derived from WavLM-large features.

## Quick Start

```bash
# Install dependencies
make setup

# Run smoke tests
make smoke

# Full reproduction (data → ASR → features → baselines → probe → eval → report)
make reproduce
```

## Project Structure

```
configs/          — YAML configurations
scripts/          — Runnable pipeline scripts
src/accentedness_routing/
  data/           — EdAcc loading & speaker-disjoint splits
  asr/            — mlx-whisper transcription & WER
  features/       — WavLM feature extraction
  triggers/       — Routing score producers (oracle, random, confidence, probe)
  routing/        — Threshold sweep & operating curves
  eval/           — Evaluation, slicing, plots
  flywheel/       — Drift detection & hard-case mining
tests/            — Unit & integration tests
experiments/      — Experiment reports & artifacts
writeup/          — Final deliverable
docs/             — Research notes & knowledge base
```

## Pipeline Steps

1. **Data** (`make data`): Load EdAcc, select 6 accents × 150 utterances, create speaker-disjoint splits
2. **ASR** (`make asr`): Transcribe with both Whisper models, compute per-utterance WER
3. **Features** (`make features`): Extract WavLM-large hidden states
4. **Baselines** (`make baselines`): Compute oracle, random, confidence, and argmax routing curves
5. **Probe** (`make probe`): Train the accentedness scalar probe
6. **Eval** (`make eval`): Generate operating curves, per-accent analysis, diagnostics
7. **Report** (`make report`): Produce final figures and summary

## Requirements

- Python 3.10-3.12
- Apple Silicon Mac (for mlx-whisper)
- ffmpeg
- ~24 GB RAM recommended
