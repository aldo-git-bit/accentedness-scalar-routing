# Accentedness-Scalar Routing for English ASR

Route English ASR utterances between a fast default recognizer and a slower careful one using a learned continuous difficulty scalar derived from WavLM-large features. The routing decision is scored by net word error rate against escalation rate — a quality-versus-cost operating curve — not by classification accuracy.

## The Core Thesis

This project reframes accent detection as an **instrument** for a **routing decision**, not as an end in itself. The accent label has no intrinsic product value; the value is the change in a downstream outcome (WER) produced by acting on it. Two interlocking problems motivate the design:

**Problem 1 — The gradient signal.** Accent is not a discrete label. It is a gradient, feature-specific property: a speaker may carry native vowels with second-language prosody, or shade continuously from lightly to heavily accented. A discrete argmax classifier forces hard categories onto a continuous phenomenon and cannot express "somewhat accented" — precisely the region where routing decisions are most delicate. A continuous difficulty scalar matches the phenomenon and doubles as a general quality/drift signal.

**Problem 2 — The unbounded taxonomy.** Discrete accent inventories break under distribution shift. Real deployments field far more accents than any fixed label set can enumerate, and N-way classification degrades sharply as N grows. A scalar's label-free nature absorbs unseen accents as a side benefit: it never enumerates a taxonomy, so it cannot fall off the end of one.

The routing system architecture is a shared front end (WavLM features computed once) fanning out to a trigger (the learned scalar, or any baseline) and to ASR. The trigger scores each utterance; above a threshold, the utterance is escalated to the careful model. Sweeping the threshold traces the operating curve. Misrouting cost is automatic: escalating a negative-gain utterance raises net WER; failing to escalate a positive-gain one leaves it unreduced.

## Key Results

Across three rounds and twelve experiments, the central finding is that the **binding constraint is headroom** — the gap between default and careful model quality — not trigger cleverness.

### Consolidated significant findings

1. **The pilot was contaminated by hallucination.** Uncapped WER (Whisper repetition loops producing WER > 1.0) dominated evaluation. Capping WER at 1.0 *reversed* the pilot's conclusions: the pilot probe fell below random, and confidence — which the pilot reported as "worse than random" — became the strongest non-oracle trigger.

2. **Free confidence captures ~47% of the oracle prize, with no training.** On the original pairing (small -> large-v3), Whisper's `avg_logprob` achieves area-vs-random of 0.0147 against an oracle ceiling of 0.0311. No learned trigger significantly beats it at this gap.

3. **The routing signal is sparse.** Median capped escalation gain is 0 for every accent. Only 47/179 test utterances have gain > 0.05. The gain concentrates in Indian English (mean 0.123); US English is slightly *negative* (-0.017).

4. **A little accent supervision helps; forcing discreteness hurts.** The multitask probe (lambda=0.1) is the best learned trigger (area 0.0108, CI excludes zero). Pushing lambda higher degrades routing as the scalar collapses toward accent identity — direct evidence for the gradient framing.

5. **Headroom is manufacturable.** Widening the model gap (tiny -> large-v3) nearly doubles the oracle prize (0.031 -> 0.058) and more than doubles the number of routing-relevant utterances (47 -> 98).

6. **Confidence degrades at wide gaps; a learned trigger crosses it.** Confidence's share of oracle headroom falls from 47% to 16% as the default model weakens. At tiny -> large-v3, the champion probe (0.014) crosses confidence (0.009) for the first time — though not yet significantly on 179 test utterances.

7. **High classification AUC does not imply better routing.** The composite combiner (AUC 0.78) and temporal-std variant (+11 AUC points) both *lose* on area-vs-random. AUC is not a reliable routing proxy.

8. **Turbo ~= large-v3 as careful model.** large-v3-turbo provides nearly identical headroom at lower cost — a selective-spend win independent of trigger quality.

### Headroom grid (Round 3)

| Pairing | WER Gap | Oracle AVR | Confidence AVR | Champion AVR | n(gain>0.05) |
|---------|---------|------------|----------------|--------------|--------------|
| small -> large-v3 | 0.043 | 0.031 | **0.015** | 0.008 | 47 |
| small -> turbo | 0.046 | 0.038 | **0.015** | 0.011 | 40 |
| base -> large-v3 | 0.120 | 0.049 | **0.018** | 0.004 | 84 |
| base -> turbo | 0.124 | 0.052 | **0.015** | -0.002 | 84 |
| tiny -> large-v3 | 0.161 | 0.058 | 0.009 | **0.014** | 98 |
| tiny -> turbo | 0.165 | 0.061 | 0.005 | -0.001 | 94 |

### Round 2 leaderboard (small -> large-v3, capped WER)

| Rank | Trigger | Area vs Random | Significant? | Source |
|------|---------|----------------|--------------|--------|
| 1 | Oracle | 0.0311 | Yes | Upper bound |
| 2 | **Confidence** | **0.0147** | **Yes** | Whisper avg_logprob |
| 3 | Multitask (lambda=0.1) | 0.0108 | Yes | Learned |
| 4 | Argmax accent | 0.0103 | No | Discrete taxonomy |
| 5 | Hallucination union | 0.0075 | No | Heuristic |
| 6 | No-speech prob | 0.0069 | No | Whisper metadata |
| 7 | Probe (gain target) | 0.0058 | No | Learned |
| 8 | Probe (capped WER) | 0.0055 | No | Learned |
| 9 | Multitask (lambda=0.0) | 0.0047 | No | Learned |
| 10 | Scalar probe (pilot) | 0.0035 | No | Learned |
| -- | Random | 0.0000 | -- | Baseline |

### The three-round arc

| Round | Question | Answer |
|-------|----------|--------|
| Pilot | Can a learned scalar route better than heuristics? | Inconclusive — evaluation contaminated by hallucination |
| Round 2 | What is the binding constraint? | **Headroom** — oracle ceiling too low, target too sparse for triggers to differentiate |
| Round 3 | Does more headroom help? | Oracle grows, confidence's share shrinks, learned trigger crosses confidence at widest gap |

### Production takeaway

1. **Narrow model gap** -> use free confidence. It captures ~half the oracle prize with no training.
2. **Wide model gap** -> a learned or composite trigger can earn its keep, but confirm on a larger test set.
3. **Prefer turbo as the careful path** — near-identical headroom at lower cost.
4. **Do not trust AUC as a routing proxy** — optimize on area-vs-random (actual WER reduction).

## Quick Start

```bash
# Install dependencies (requires uv)
make setup

# Run tests (38 tests: smoke, E2E, eval functions)
make test

# Full reproduction — Round 2 (from cached ASR + features)
make reproduce

# Full reproduction — Round 3 (decodes 3 new models, then evaluates)
make reproduce-v3
```

## Project Structure

```
configs/          - YAML configurations (default.yaml with asr_grid, combiner, etc.)
scripts/          - Runnable pipeline scripts (per-phase and per-experiment)
src/accentedness_routing/
  data/           - EdAcc loading & speaker-disjoint splits
  asr/            - mlx-whisper transcription, WER, caching
  features/       - WavLM feature extraction (mean-pool and stats variants)
  triggers/       - Routing score producers (oracle, random, confidence, probe,
                    hallucination, multitask, combiner)
  routing/        - Threshold sweep & operating curves
  eval/           - Shared evaluation harness (eval_common.py), slicing, plots
  flywheel/       - Drift detection & hard-case mining
tests/            - Unit, integration, and eval-function tests
experiments/      - Experiment reports, metrics.json, figures, COMPARISON.md
writeup/          - Writeups per round (v1 pilot, v2 Round 2, v3 Round 3,
                    v4-detailed consolidated)
docs/             - Research notes, knowledge base, decisions log, AI use log
models/           - Trained probe checkpoints
```

## Pipeline Steps

### Pilot + Round 2 (original pairing: small -> large-v3)

| Step | Command | Description |
|------|---------|-------------|
| 1 | `make data` | Load EdAcc, select 6 accents x 150 utterances, speaker-disjoint splits |
| 2 | `make asr` | Transcribe with Whisper-small and Whisper-large-v3 |
| 3 | `make features` | Extract WavLM-large hidden states (25 layers x 1024 dims) |
| 4 | `make baselines` | Oracle, random, confidence, argmax-accent operating curves |
| 5 | `make probe` | Train accentedness scalar probe |
| 6 | `make eval` | Operating curves, per-accent analysis, speaker leakage check |
| 7 | `make rescore` | Re-score pilot with capped WER + bootstrap CIs (EXP-00) |
| 8 | `make ext1` - `make ext5` | Round 2 extensions (gain target, diagnostics, hallucination, multitask) |

### Round 3 (headroom grid: {tiny,base,small} x {turbo,large-v3})

| Step | Command | Description |
|------|---------|-------------|
| 1 | `make asr-ladder` | Decode all models in the grid (tiny, base cached; turbo new) |
| 2 | `make acoustic-features` | Extract duration, silence ratio, speaking rate per utterance |
| 3 | `make ext-headroom` | Headroom grid eval (EXP-09) + headroom sweep figure |
| 4 | `make ext-composite` | Composite trigger eval (EXP-10) |
| 5 | `make features-stats` | Extract WavLM stats features (mean + std per layer) |
| 6 | `make ext-temporal` | Temporal std eval (EXP-11) |
| 7 | `make compare` | Regenerate COMPARISON.md and all figures |

All steps cache their outputs. Re-running a completed step returns instantly.

## Experiment Index

| EXP | Round | Description | Status |
|-----|-------|-------------|--------|
| 00 | R2 | Re-score pilot with capped WER + bootstrap CIs | Complete |
| 01 | Pilot | Pipeline scaffold (data -> ASR -> features -> probe) | Complete |
| 02 | Pilot | Scalar probe vs baselines (contaminated — see writeup) | Complete (invalid) |
| 03 | Pilot | Flywheel: drift detection & hard-case mining | Complete |
| 04 | R2 | Gain-target probe (target fix) | Complete |
| 05 | R2 | Learning curve + accent classifier diagnostics | Complete |
| 06 | R2 | Hallucination baselines + confidence autopsy | Complete |
| 07 | R2 | Stats pooling infrastructure (deferred, realized as EXP-11) | Deferred |
| 08 | R2 | Multi-task probe (lambda sweep) | Complete |
| 09 | R3 | Headroom grid (6 model pairings) | Complete |
| 10 | R3 | Composite trigger (logistic combiner) | Complete |
| 11 | R3 | Temporal std (WavLM per-layer variability) | Complete |
| 12 | R3 | Accent-adapted careful path (stretch, gated) | Deferred |

The living leaderboard is at `experiments/COMPARISON.md` (66 entries). The headroom sweep figure is at `experiments/figures/headroom_sweep.png`.

## Dataset

[EdAcc](https://huggingface.co/datasets/edinburghcstr/edacc) (Edinburgh International Accents of English) — conversational English dyads with self-reported accents.

6 accents x 150 utterances = 900 total. Speaker-disjoint splits: 544 train / 177 val / 179 test across 49 speakers.

### Per-accent gain structure (capped WER, small -> large-v3)

| Accent | n (test) | Capped Default WER | Mean Capped Gain | Routing Helps? |
|--------|----------|--------------------|------------------|----------------|
| Indian English | 35 | 0.50 | **0.123** | Yes (the prize) |
| US English | 27 | 0.45 | -0.017 | No (careful sometimes worse) |
| Southern British | 28 | 0.38 | small + | Marginal |
| Scottish English | 37 | 0.30 | ~0 | No |
| Nigerian English | 30 | 0.36 | ~0 | No |
| Irish English | 22 | 0.31 | ~0 | No |

Median capped gain is 0 for every accent. The positive routing prize concentrates in Indian English.

## Probe Architecture

```
WavLM-large (25 layers x 1024) -> mean-pool per layer
  -> LearnableWeightedSum(25) -> Linear(1024, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 1)
```

- ~263K trainable parameters
- **Pilot target:** per-utterance WER from Whisper-small (contaminated by hallucination — see writeup)
- **Round 2+ target:** capped escalation gain = `cap_wer(default) - cap_wer(careful)`
- Loss: HuberLoss(delta=0.1) — bounds outlier gradients, but cannot save a contaminated target
- Multi-task variant (Round 2): adds an accent classification head with weight lambda; lambda=0.1 is best
- Early stopping on validation loss (patience=10)
- Score calibration: percentile normalization from training set

## Models

| Model | Role | Framework |
|-------|------|-----------|
| Whisper tiny/base/small | Default path (grid) | mlx-whisper |
| Whisper large-v3 | Careful path | mlx-whisper |
| Whisper large-v3-turbo | Cheaper careful path | mlx-whisper |
| WavLM-large | Feature extractor (25 layers x 1024) | PyTorch (CPU) |
| CommonAccent XLSR | Argmax-accent baseline | PyTorch |

## Requirements

- Python 3.10-3.12
- Apple Silicon Mac (for mlx-whisper)
- ffmpeg
- ~24 GB RAM recommended
- [uv](https://docs.astral.sh/uv/) package manager

## Tests

```bash
make test       # All 38 tests (smoke, E2E, eval functions)
make smoke      # Smoke tests only (WavLM, mlx-whisper, jiwer)
```

## Detailed Writeup

The consolidated technical report covering all three rounds is at [`writeup/writeup-v4-detailed.md`](writeup/writeup-v4-detailed.md). Per-round writeups are also available (v1 pilot, v2 Round 2, v3 Round 3).

## AI Use

This project was implemented with assistance from Claude Code (Anthropic). All experimental design, analysis, and interpretation were guided by pre-approved research plans with pre-registered decision rules. AI assistance was used for code generation, debugging, and drafting. The human owned the framing, metric integrity, decision rules, and judgment about which results to believe. See `docs/ai-use-log.md` for details.
