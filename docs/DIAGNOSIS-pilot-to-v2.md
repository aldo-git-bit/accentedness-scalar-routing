# Diagnosis: Pilot to Round 2

Critical assessment of the pilot's weak signal (Pearson r ~ 0.21) and the plan to resolve whether the bottleneck is the **target**, the **data**, or the **architecture**.

## 1. WER Contamination / Hallucination Artifact

The pilot used **uncapped WER** as both the training target and the evaluation metric. Whisper-small produces hallucinations (repetitive or invented text) on some utterances, yielding WER values of 3x-9x. These outliers:

- **Inflate mean WER** disproportionately. A single hallucinated utterance (WER = 8.96) can swing the mean of a 30-utterance accent group by ~0.27.
- **Distort Pearson r.** A few extreme residuals dominate the correlation, making the metric uninterpretable for the bulk of the data. The pilot's r ~ 0.21 may be measuring "how well does the probe predict hallucination?" rather than "how well does it predict routing benefit?"
- **Corrupt the training target.** The probe is trained to regress toward WER values that are artifacts of model failure, not genuine difficulty signals.

**Resolution:** `eval_common.cap_wer()` clamps all WER values at 1.0 before any computation. Phase 0 re-scores the pilot to quantify how much this changes the story.

## 2. Effective Signal Density

The dataset has 900 utterances total, ~544 with complete ASR cache, ~179 in the test fold. But the **effective signal density** is much lower:

- Many utterances show near-zero escalation gain (careful model is not meaningfully better).
- Hallucinated utterances (WER > 1.0) are noise, not signal. After capping, their gain is bounded at 1.0 - cap_wer(careful), which may still be large but is at least bounded.
- The actual number of utterances where routing matters (gain > 0.05) may be much smaller than 179.

This means the test fold may have **insufficient statistical power** for fine-grained comparisons. Bootstrap CIs will make this visible.

## 3. Why "Just Add Cross-Validation" Isn't the First Move

With speaker-disjoint splits and only ~15-25 speakers per accent, k-fold CV would:
- Require re-splitting speakers (expensive, complicates reproducibility).
- Not address the fundamental question of whether the target is wrong.
- Mask the real problem: if the probe is learning the wrong thing, more folds of the wrong thing don't help.

The right sequence is: fix the target first (Ext 1), then assess data limitations (Ext 2), then consider architectural changes (Ext 4-5).

## 4. Metric Critique

### Pearson r on spike+tail distributions

The pilot reported Pearson r between probe scores and default-model WER. This metric is problematic because:

- WER has a spike at 0 (many utterances are transcribed perfectly) and a heavy tail (hallucinations).
- Pearson r assumes a bivariate normal distribution and is sensitive to outliers.
- A "good" Pearson r could be achieved by a model that only predicts hallucination vs. non-hallucination, which is not useful for routing.

**Better alternatives (implemented in eval_common):**
- `decision_scorecard`: AUC and AP for the binary event gain > tau, which directly measures routing utility.
- `operating_curve` with capped WER: evaluates the actual WER reduction at each escalation budget.
- Spearman r: rank correlation, robust to non-normality.

## 5. Three-Way Uncertainty Mapping

The pilot's weak signal could be caused by three distinct bottlenecks:

| Bottleneck | Hypothesis | Extension | Diagnostic |
|-----------|-----------|-----------|-----------|
| **Target** | Training on uncapped WER teaches the probe to predict hallucination, not routing benefit | Ext 1 | gain_target probe dominates capped_wer and pilot? |
| **Data** | 544 utterances (60% train) is too few; probe is data-limited | Ext 2 | Learning curve still climbing at 100%? |
| **Architecture** | Mean-pooled WavLM features lose variance information needed to detect difficulty | Ext 4 | mean+std probe beats mean-only? |

**Additional diagnostics:**
- Ext 3 (hallucination baselines) tests whether cheap hallucination detectors explain the routing signal.
- Ext 5 (multi-task) tests whether accent supervision regularizes the probe.

## 6. Decision Framework

After Phase 0 re-scoring:

1. If capped WER **does not change** the pilot's story: the problem is not hallucination contamination. Proceed directly to Ext 1 to test the target hypothesis.
2. If capped WER **substantially changes** the story (e.g., confidence trigger improves, oracle ceiling drops): hallucination was a major confound. Ext 3 becomes higher priority.

After Ext 1:
- If gain_target probe dominates: the target was the bottleneck. Use gain probe as champion for Ext 2/4/5.
- If gain_target probe ≈ capped_wer probe: the target is not the bottleneck. Data or architecture is.
- If both new probes ≈ argmax_accent: the scalar adds nothing beyond accent identity. Architectural dead-end for this feature set.
