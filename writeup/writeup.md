# Accentedness-Scalar Routing for English ASR

## Problem Definition

Production ASR systems face a fundamental trade-off between accuracy and computational cost. Accented speech disproportionately degrades smaller, faster models while larger models handle it well — but routing every utterance to the large model is prohibitively expensive. We need a per-utterance routing decision: *when* is it worth escalating from a fast default recognizer to a careful one?

This work proposes a **learned scalar trigger** that predicts per-utterance difficulty from acoustic features, enabling cost-aware routing between Whisper-small (default) and Whisper-large-v3 (careful).

## Baseline System

We evaluate against four baselines on the EdAcc dataset (6 English accents × 150 utterances, 900 total):

1. **Oracle** — routes based on true WER improvement (upper bound)
2. **Random** — uniform random escalation (no-information baseline)
3. **Confidence** — routes based on Whisper-small's avg_logprob (lower confidence → escalate)
4. **Argmax Accent** — routes entire accent groups based on group-level mean WER

All baselines produce a scalar score per utterance. We sweep thresholds to generate **operating curves** (net WER vs. escalation rate), enabling fair comparison at any escalation budget.

## Improvement: Accentedness Scalar Probe

Our contribution is a lightweight probe (~263K parameters) trained on WavLM-large representations:

- **Input**: Per-layer mean-pooled hidden states from WavLM-large (25 layers × 1024 dims)
- **Architecture**: LearnableWeightedSum → Linear(1024, 256) → ReLU → Dropout(0.1) → Linear(256, 1)
- **Target**: Per-utterance WER from the default model (Whisper-small)
- **Loss**: HuberLoss(δ=0.1) for robustness to high-WER outliers
- **Training**: Speaker-disjoint splits (544 train / 177 val / 179 test); AdamW(lr=1e-3); early stopping on validation loss (patience=10)
- **Score calibration**: Percentile normalization (2nd/98th) from training set predictions, clipped to [0, 1]

The probe learns to weight WavLM layers and predict utterance difficulty directly, producing a continuous score that captures more fine-grained difficulty variation than the discrete argmax-accent baseline.

## Results

### Operating Curve Comparison (test set, 179 utterances)

| Trigger | WER @10% | WER @20% | WER @30% | WER @50% | Area vs Random |
|---------|----------|----------|----------|----------|----------------|
| Oracle | 0.76 | 0.74 | 0.73 | 0.73 | 0.65 |
| **Scalar probe** | **1.72** | **1.00** | **0.80** | **0.78** | **0.45** |
| Argmax accent | N/A | 1.28 | 1.08 | 0.78 | 0.45 |
| Random | 2.45 | 2.44 | 1.21 | 1.20 | 0.00 |
| Confidence | 2.94 | 2.92 | 2.92 | 2.91 | -1.33 |

The scalar probe outperforms the argmax-accent baseline at the 20% and 30% escalation budgets (1.00 vs 1.28 and 0.80 vs 1.08 WER, respectively) and matches it at 50%. Area-vs-random is comparable (0.45 for both). However, the probe **does not strictly dominate** argmax at all thresholds — their curves cross at certain escalation rates.

### Per-Accent WER Analysis

| Accent | Default WER | Careful WER | Escalation Gain |
|--------|-------------|-------------|-----------------|
| Indian English | 8.96 | 0.40 | 8.56 |
| US English | 5.14 | 3.13 | 2.01 |
| Southern British | 1.67 | 0.35 | 1.32 |
| Scottish English | 0.33 | 0.26 | 0.07 |
| Irish English | 0.31 | 0.29 | 0.02 |
| Nigerian English | 0.40 | 0.39 | 0.01 |

The routing problem is concentrated on 2-3 accents: Indian English (8.56 gain), US English (2.01), and Southern British English (1.32). For Irish, Scottish, and Nigerian English, both models perform similarly well, so routing provides negligible benefit. This concentration partly explains why argmax accent is a strong baseline — simply always escalating Indian and US English captures most of the available gain.

### Probe Training

The probe trained for 14 epochs before early stopping (best at epoch 4, val loss = 0.271). Peak validation Pearson r was ~0.21, indicating a weak but nonzero correlation between the predicted scalar and actual utterance-level WER.

### WavLM Layer Weights

Learned layer weights were nearly uniform across all 25 layers (~0.041 each), meaning no single WavLM layer disproportionately encodes the difficulty signal. This suggests the information relevant to predicting ASR difficulty is distributed across the model's depth rather than concentrated in specific layers. The LearnableWeightedSum effectively reduces to mean pooling, and the downstream MLP selects relevant dimensions from the averaged 1024-dim representation.

A limitation of the weighted-sum approach: if different layers encode complementary aspects of difficulty (e.g., phonetic confusion vs. syntactic complexity), averaging may blur those signals. More expressive pooling strategies (e.g., attention over layers, per-layer projection heads) could potentially improve performance but would require more training data to avoid overfitting.

### Speaker Leakage

- MI(score, speaker) = 0.32
- MI(score, accent) = 0.22
- Ratio = 1.44

The probe's scalar correlates more with speaker identity than accent identity. This is not necessarily a leak — utterance-level difficulty genuinely varies by speaker, and the probe is capturing that variation. However, it means the scalar is partially a speaker recognizer rather than a pure difficulty estimator, which may limit generalization to unseen speakers.

### Hard-Case Mining

The flywheel analysis identified extreme misrouting-cost utterances where Whisper-small produces catastrophic errors (WER > 50) while Whisper-large handles them correctly. These cases — likely involving hallucination or language confusion — represent the highest-value routing targets and are correctly assigned high probe scores.

## Failure Modes & Limitations

1. **Small speaker count**: EdAcc has only 4-17 speakers per accent. Speaker-disjoint splits leave very few speakers in val/test (9 each), limiting statistical power and making results sensitive to which speakers end up in which fold.

2. **Weak learned signal**: Val Pearson r of ~0.21 is below the 0.3 target. The probe has limited predictive power for per-utterance WER, though this is still sufficient to improve routing decisions at practical escalation budgets.

3. **Scalar target choice**: We regress default-model WER, not escalation gain (which would require careful-model transcripts during training). This works when WER correlates with gain, but utterances where both models fail equally receive high scores without routing benefit.

4. **Concentrated routing benefit**: Escalation gain is dominated by 2-3 accents. The scalar probe's advantage over argmax is marginal because group-level routing already captures most of the value.

5. **Feature extraction cost**: WavLM-large inference is itself computationally expensive. In production, a lighter feature extractor or distilled model would be needed to make routing cost-effective.

6. **Confidence baseline failure**: Whisper's avg_logprob is worse than random for routing — the model's internal confidence does not reflect difficulty on accented speech. This is a notable negative finding.

## Open Questions

- Would multi-task training (WER regression + accent classification) improve the scalar's discriminative power?
- Can the probe generalize to accents not seen during training?
- What is the break-even point where feature extraction cost exceeds ASR cost savings?
- Would a probe trained directly on escalation gain (requiring careful-model transcripts) outperform the WER-regression proxy?
- Could concatenation of per-layer representations (25 × 1024 → MLP) outperform the weighted-sum approach, given sufficient training data?

## Reproduction

```bash
git clone <repo-url> && cd accentedness-scalar-routing
make setup          # install dependencies via uv
make smoke          # verify WavLM, mlx-whisper, jiwer
make reproduce      # full pipeline: data → asr → features → baselines → probe → eval → report
```

Results are saved to `experiments/`. The operating curve figure is at `experiments/EXP-02-scalar-vs-baselines/figures/operating_curves.png`.

## AI Use

This project was implemented with assistance from Claude Code (Anthropic). All architectural decisions, experimental design, and analysis were guided by the research plan; AI assistance was used for code generation and debugging. See `docs/ai-use-log.md` for details.
