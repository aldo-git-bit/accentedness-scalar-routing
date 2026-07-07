# Round 3: Finding the Regime Where Trigger Quality Matters

## Motivation

Round 2 established that the binding constraint is **headroom**, not the trigger. The oracle area-vs-random was only 0.031 (small-vs-large-v3), confidence captured 47% of it for free, and the remaining ~0.016 was smaller than the CI width (~0.012). Round 3 asks: if we manufacture more headroom by widening the model gap, does a regime emerge where learned triggers separate from confidence?

Three levers:

1. **Widen the gap.** Decode with tiny and base as default models (not just small) against both large-v3 and large-v3-turbo as careful models, creating 6 pairings with varying WER gaps.
2. **Composite signals.** Combine confidence, no-speech probability, champion probe score, and acoustic features (duration, silence ratio, speaking rate) via logistic regression.
3. **Temporal information.** Add WavLM temporal variability (std-pooled features) to the combiner.

## Phase 0: Infrastructure

The Round 2 evaluation harness (`eval_common.py`) was extended with three functions:

- **`headroom_summary`** — computes oracle ceiling, WER gap, and gain distribution statistics per pairing
- **`combiner_eval`** — fits L2 logistic regression on validation features, scores test, runs paired bootstrap vs confidence
- **`grid_eval`** — loops over multiple model pairings with a shared evaluation pipeline

Configuration was extended with `asr_grid` (3 default models x 2 careful models), `combiner` (tau, family), and `acoustic_features` sections. Unit tests cover all new functions (38 tests passing).

## EXP-09: Headroom Grid

Six pairings were evaluated, spanning WER gaps from 0.043 to 0.165:

| Pairing | WER Gap | Oracle AVR | Confidence AVR | Champion AVR | n(gain > 0.05) |
|---------|---------|------------|----------------|--------------|----------------|
| small -> large-v3 | 0.043 | 0.031 | **0.015** | 0.008 | 47 |
| small -> turbo | 0.046 | 0.038 | **0.015** | 0.011 | 40 |
| base -> large-v3 | 0.120 | 0.049 | **0.018** | 0.004 | 84 |
| base -> turbo | 0.124 | 0.052 | **0.015** | -0.002 | 84 |
| tiny -> large-v3 | 0.161 | 0.058 | 0.009 | **0.014** | 98 |
| tiny -> turbo | 0.165 | 0.061 | 0.005 | -0.001 | 94 |

### Key findings

**1. Oracle headroom grows with gap.** Oracle area-vs-random nearly doubles from 0.031 (small->large-v3) to 0.058-0.061 (tiny->large-v3/turbo). The headroom constraint is real and manufacturable.

**2. Confidence degrades at wide gaps.** At the narrow gap (small->large-v3), confidence captures 47% of oracle headroom (0.015/0.031). At the widest gap (tiny->large-v3), it captures only 16% (0.009/0.058). Confidence's logprob comes from the small/base/tiny default model — as the default model gets weaker, its confidence scores become less informative about when escalation helps.

**3. The champion probe inverts at wide gaps.** The retrained multitask lambda=0.1 probe outperforms confidence at tiny->large-v3 (0.014 vs 0.009) but underperforms at narrow gaps. This is the first evidence of a regime where a learned trigger beats confidence.

**4. Turbo ~= large-v3 as careful model.** The two careful models produce nearly identical headroom at each gap level (e.g., 0.052 vs 0.049 at base tier). Turbo is a viable cheaper careful path.

**5. Signal density grows with gap.** At narrow gaps, only 47/179 utterances have gain > 0.05. At wide gaps, 94-98 do. Wider gaps don't just increase headroom — they increase the number of utterances where routing matters, making the learning problem tractable.

The headroom sweep figure (`experiments/figures/headroom_sweep.png`) shows oracle area growing linearly with WER gap, confidence flattening, and the champion probe crossing confidence at the widest gaps.

## EXP-10: Composite Trigger

A composite trigger combining [confidence, no_speech_prob, champion_score, duration, silence_ratio, speaking_rate] via L2 logistic regression was tested under two regimes:

| Regime | Val AUC | Test AUC | Area vs Random | Confidence AVR |
|--------|---------|----------|----------------|----------------|
| Narrow (small->large-v3) | 0.688 | 0.654 | 0.005 | 0.019 |
| Wide (tiny->large-v3) | 0.737 | 0.776 | 0.007 | 0.020 |

The composite trigger achieves substantially higher AUC at the wide gap (0.776 vs 0.654), confirming that the model can discriminate "gain > 0.05" utterances better when signal density is high. However, the operating-curve area-vs-random metric — which measures actual WER reduction, not just classification — shows the composite underperforming confidence in both regimes.

This AUC-vs-area disconnect reveals an important subtlety: discriminating high-gain utterances (AUC) is not the same as routing in a way that reduces WER (area). The composite correctly identifies which utterances would benefit from escalation but doesn't translate that into proportionally better WER at low escalation rates, possibly because it's overly confident on some easy utterances that don't need escalation.

## EXP-11: Temporal Std

WavLM stats features (mean + std across time per layer) were extracted for all 900 utterances. A scalar "temporal variability" feature (mean of per-layer std values) was added to the composite combiner and tested on the wide gap (tiny->large-v3) regime.

| Combiner | Val AUC | Test AUC | Area vs Random | CI |
|----------|---------|----------|----------------|-----|
| Without temporal_std | 0.634 | 0.672 | 0.021 | [0.006, 0.037] |
| **With temporal_std** | **0.745** | **0.786** | **0.024** | **[0.009, 0.040]** |
| Paired diff (with - without) | — | — | +0.003 | [-0.005, +0.011] |

Temporal_std is the **largest single feature coefficient** in the combiner (1.34, vs 0.87 for champion_score and -0.25 for confidence). It substantially improves classification AUC (+11 points on test), but the area-vs-random improvement (+0.003) is not statistically significant (CI includes zero).

### Confidently-hallucinated subset

A subset analysis on utterances with capped WER > 0 and above-median avg_logprob (85/179 test utterances) — cases where confidence fails — shows no temporal_std advantage. The without-temporal combiner actually slightly outperforms on this subset (area 0.023 vs 0.019). Temporal variability does not specifically help on confidence's blind spot.

### Interpretation

Temporal_std captures something real (the AUC improvement is substantial) but it correlates with overall difficulty rather than with the specific failure mode where escalation helps. It's not adding information about *when the careful model would do better* — it's adding information about *how hard the utterance is*, which is already partially captured by confidence and the other features.

## Decision Rules: What We Pre-Registered

| Rule | Outcome |
|------|---------|
| Oracle area grows with gap | **Confirmed.** 0.031 -> 0.058 (1.9x) |
| Confidence keeps most headroom at all gaps | **Rejected at wide gaps.** Confidence captures 47% at narrow but only 16% at wide |
| A learned trigger separates from confidence past some gap | **Partially confirmed.** Champion probe crosses confidence at tiny->large-v3 but not consistently |
| Turbo ~= large-v3 headroom at lower cost | **Confirmed.** Nearly identical headroom |
| Composite beats confidence at wide gap | **Rejected.** Higher AUC but lower area-vs-random |
| Temporal_std helps on confidently-hallucinated subset | **Rejected.** No improvement on confidence's blind spot |

## Synthesis

### What Round 3 adds to the narrative

Round 2 ended with a clear diagnosis: the binding constraint is headroom, not the trigger. Round 3 confirms this by manipulating headroom and observing what happens.

**The good news:** Headroom is manufacturable. Widening the model gap from small->large-v3 (WER gap = 0.043) to tiny->large-v3 (WER gap = 0.161) nearly doubles the oracle ceiling and more than doubles the number of routing-beneficial utterances. At the wide gap, a learned trigger (champion probe, area = 0.014) outperforms confidence (area = 0.009) for the first time.

**The bad news:** Even at the widest gap, the advantage is fragile. The composite trigger achieves impressive classification AUC (0.78) but doesn't translate into WER reduction that beats confidence. Temporal features improve discrimination further but don't improve routing. The gap between "correctly identifying hard utterances" and "routing in a way that reduces WER" persists.

**The core problem:** Confidence's area-vs-random doesn't scale with headroom — it captures a roughly constant absolute amount (~0.015) regardless of the model gap. The oracle ceiling grows but confidence doesn't keep up. Yet no learned trigger consistently fills the gap either. The residual headroom (oracle minus confidence) grows from 0.016 to 0.049, but learned triggers capture only a fraction of it.

### Three-round arc

| Round | Question | Answer |
|-------|----------|--------|
| Pilot | Can a learned scalar route better than heuristics? | Inconclusive (contaminated evaluation) |
| Round 2 | What is the binding constraint? | Headroom — oracle ceiling is too low for triggers to differentiate |
| Round 3 | Does more headroom help? | Oracle grows, confidence weakens, learned triggers improve but inconsistently |

### Practical implications

1. **For narrow model gaps (small -> large-v3):** Use confidence. It captures nearly half the oracle headroom, is free, and no learned trigger significantly beats it.

2. **For wide model gaps (tiny -> large-v3):** Confidence degrades. A retrained champion probe or composite trigger can outperform it, but the advantage is not yet statistically robust on 179 test utterances. Scaling the test set would clarify whether the champion's advantage is real.

3. **Turbo as cheap careful path:** large-v3-turbo provides nearly identical headroom to large-v3 at lower cost. This makes the routing decision more economically attractive.

4. **Temporal features are informative but not actionable.** WavLM temporal variability improves AUC by 11 points but doesn't improve routing WER. The feature captures general difficulty, not escalation-specific difficulty.

## Open questions

1. **Sample size.** The champion probe's advantage at wide gaps (area 0.014 vs 0.009) is not significant on 179 test utterances. A 5-10x larger test set with similar accent diversity could resolve this.

2. **AUC-area disconnect.** Why does high classification AUC not translate to area-vs-random? Is it threshold sensitivity, miscalibration, or something about the operating-curve geometry?

3. **Accent-adapted careful paths (EXP-12, deferred).** An Indian-accent finetuned model could expand headroom specifically where it matters most. This was gated behind a verify-before-trust guard and not run.

4. **Beyond mean-pooled WavLM.** The temporal_std result suggests mean+std pooling captures general difficulty but misses escalation-specific patterns. Attention-weighted or segment-level features could do better.

## Reproduction

```bash
git clone <repo-url> && cd accentedness-scalar-routing
make setup

# Phase 0: infrastructure (already run if you have Round 2)
make rescore && make diagnose

# Phase 1: headroom grid (decodes tiny, base, turbo; ~30 min on M-series)
make asr-ladder
make acoustic-features
make ext-headroom

# Phase 2: composite trigger
make ext-composite

# Phase 3: temporal std (requires WavLM re-extraction; ~1 hr CPU)
make features-stats
make ext-temporal

# Final comparison
make compare
```

Results are in `experiments/`. The master comparison is at `experiments/COMPARISON.md`. The headroom sweep figure is at `experiments/figures/headroom_sweep.png`.

## AI Use

This project was implemented with assistance from Claude Code (Anthropic). All experimental design, analysis, and interpretation were guided by a pre-approved research plan. AI assistance was used for code generation, debugging, and drafting. See `docs/ai-use-log.md` for details.
