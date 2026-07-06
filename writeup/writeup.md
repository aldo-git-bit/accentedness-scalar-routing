# Accentedness-Scalar Routing for English ASR

## Problem Definition

Production ASR systems face a fundamental trade-off between accuracy and computational cost. Accented speech disproportionately degrades smaller, faster models while larger models handle it well — but routing every utterance to the large model is prohibitively expensive. We need a per-utterance routing decision: *when* is it worth escalating from a fast default recognizer to a careful one?

This work proposes a **learned scalar trigger** that predicts per-utterance difficulty from acoustic features, enabling cost-aware routing between Whisper-small (default) and Whisper-large-v3 (careful).

## Baseline System

We evaluate against four baselines on the EdAcc dataset (6 English accents × 150 utterances):

1. **Oracle** — routes based on true WER improvement (upper bound)
2. **Random** — uniform random escalation (no-information baseline)
3. **Confidence** — routes based on Whisper-small's avg_logprob (lower confidence → escalate)
4. **Argmax Accent** — routes entire accent groups based on group-level mean WER

All baselines produce a scalar score per utterance. We sweep thresholds to generate **operating curves** (net WER vs. escalation rate), enabling fair comparison at any escalation budget.

## Improvement: Accentedness Scalar Probe

Our contribution is a lightweight probe trained on WavLM-large representations:

- **Input**: Per-layer mean-pooled hidden states from WavLM-large (25 layers × 1024 dims)
- **Architecture**: LearnableWeightedSum → Linear(1024, 256) → ReLU → Dropout → Linear(256, 1)
- **Target**: Per-utterance WER from the default model (Whisper-small)
- **Loss**: HuberLoss(δ=0.1) for robustness to high-WER outliers
- **Training**: Speaker-disjoint splits; early stopping on validation loss

The probe learns to weight WavLM layers and predict utterance difficulty directly, producing a continuous score that captures more fine-grained difficulty variation than the discrete argmax-accent baseline.

## Evaluation Plan & Results

### Metrics
- **Operating curve**: Net WER at each escalation rate (0-100%)
- **Net WER @ 20%**: WER when escalating 20% of utterances
- **Area vs Random**: Area between trigger curve and random baseline
- **Dominance**: Whether the scalar curve is ≤ argmax at all escalation rates

### Per-Accent Analysis
We slice results by accent to check whether the probe improves routing for all accents or only a subset. We also analyze:
- Learned WavLM layer weights (which layers encode difficulty?)
- Score distributions by accent (does the scalar separate easy from hard?)
- Speaker leakage (MI between scores and speakers vs. scores and accents)

## Failure Modes & Limitations

1. **Small speaker count**: EdAcc has only 4-17 speakers per accent. Speaker-disjoint splits are valid but leave very few speakers in val/test, limiting statistical power.
2. **Scalar target choice**: We regress default-model WER, not escalation gain. This works if WER correlates with gain, but utterances where both models fail equally get high scores without routing benefit.
3. **No cross-accent generalization test**: All 6 accents appear in training. Unseen accents would test true generalization.
4. **Feature extraction cost**: WavLM-large inference is itself expensive. In production, a lighter feature extractor would be needed.

## Open Questions

- Would multi-task training (WER regression + accent classification) improve routing?
- Can the probe generalize to accents not seen during training?
- What is the break-even point where the feature extraction cost exceeds the ASR cost savings?

## AI Use

This project was implemented with assistance from Claude Code (Anthropic). See `docs/ai-use-log.md` for details. All architectural decisions, experimental design, and analysis were guided by the research plan; AI assistance was used for code generation and debugging.
