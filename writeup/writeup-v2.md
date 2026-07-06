# Round 2: From Pilot to Better Models

## Motivation

The pilot scalar probe achieved Pearson r ~ 0.21 against uncapped default-model WER — a weak signal that left open whether the bottleneck was the **target** (uncapped WER contaminated by hallucination), the **data** (544 training utterances), or the **architecture** (mean-pooled WavLM features losing temporal information). Round 2 is a systematic ablation across five extensions, sharing a single evaluation harness and a frozen 179-utterance test fold, designed to isolate the binding constraint.

## Evaluation Harness

All Round 2 experiments use `eval_common.py`, which differs from the pilot evaluation in three ways:

1. **Capped WER.** `cap_wer(w) = min(w, 1.0)` is applied to all WER values before any computation. This neutralizes hallucination artifacts where Whisper-small generates repetition loops (WER = 8.96 for Indian English, uncapped).

2. **Decision scorecard.** Beyond operating curves, every trigger is evaluated for AUC and average precision on the binary event "escalation gain > tau" (tau = 0, 0.05), plus Spearman and Pearson rank correlations against capped gain.

3. **Bootstrap CIs.** 1,000 resamples of utterance IDs, recomputing full operating curves per resample, yielding 95% confidence intervals on all summary scalars and per-threshold curve bands. Paired bootstrap tests whether trigger A significantly differs from trigger B.

## Phase 0: The Pilot Was Broken

Re-scoring the five pilot triggers with capped WER produced a dramatically different ranking:

| Trigger | Area vs Random | 95% CI | AUC (tau=0.05) |
|---------|---------------|--------|----------------|
| Oracle | 0.0311 | [0.020, 0.045] | 0.956 |
| **Confidence** | **0.0147** | **[0.003, 0.028]** | **0.656** |
| Argmax accent | 0.0103 | [-0.002, 0.023] | 0.515 |
| Scalar probe (pilot) | 0.0035 | [-0.006, 0.013] | 0.427 |
| Random | 0.0000 | [0.000, 0.000] | 0.471 |

The pilot probe performed **worse than random** under capped WER (AUC = 0.32 at tau=0, below the 0.5 chance level). It had learned to predict hallucination artifacts, not routing benefit. The pilot writeup's claim that confidence was "worse than random" (area = -1.33) was entirely an artifact of uncapped WER — under capped evaluation, confidence is the best non-oracle trigger.

### Hallucination contamination

9 of 179 test utterances (5%) have default-model WER > 1.0. The damage is concentrated:

| Accent | n | Hallucinated | Uncapped mean WER | Capped mean WER |
|--------|---|-------------|-------------------|-----------------|
| Indian English | 35 | 3 | 8.96 | 0.50 |
| US English | 27 | 3 | 5.14 | 0.45 |
| Southern British | 28 | 1 | 1.67 | 0.38 |
| Scottish | 37 | 1 | 0.33 | 0.30 |
| Nigerian | 30 | 1 | 0.40 | 0.36 |
| Irish | 22 | 0 | 0.31 | 0.31 |

Capping reduces mean default WER from 2.96 to 0.39 and mean careful WER from 0.76 to 0.34. The oracle ceiling (area = 0.031) is modest — routing headroom is genuinely limited.

### Effective signal density

Median capped escalation gain is **0.0 for every accent**. For more than half of utterances, escalating to the careful model provides zero benefit. The effective number of utterances where routing matters (gain > 0.05) is much smaller than 179, limiting statistical power for fine-grained comparisons.

## Extension 1: Target Probe (EXP-04)

**Question:** Does training on escalation gain (cap(default) - cap(careful)) instead of uncapped WER produce a better routing trigger?

Two new probes with identical architecture, different targets:

| Probe | Target | Best val loss | Val Pearson r | Test area vs random |
|-------|--------|-------------|---------------|-------------------|
| gain_target | escalation_gain | 0.009 | ~0.05 (unstable) | 0.0058 |
| capped_wer | cap_wer(default) | 0.015 | 0.675 | 0.0055 |
| pilot | uncapped WER | — | ~0.21 | 0.0035 |

The gain-target probe never stabilized during training — val Pearson r oscillated around zero and early-stopped at epoch 7. The capped-WER probe learned a much more stable regression (r = 0.675) but did not translate that into routing advantage.

**Paired bootstrap:** No probe significantly outperforms any other (all CIs on area_vs_random difference include zero). Neither dominates argmax accent.

**Verdict:** Fixing the target from uncapped WER to gain or capped WER improves training stability but does not produce a trigger that beats confidence. The target was contaminated, but fixing it was necessary, not sufficient.

## Extension 2: Learning Curve + Accent Classifier (EXP-05)

**Question:** Is the probe data-limited? Does the learned representation specialize differently from an accent classifier?

### Learning curve

The champion (gain-target) probe was trained on {25%, 50%, 75%, 100%} of train speakers:

| Fraction | n_utterances | Val AUC | Val Pearson r |
|----------|-------------|---------|---------------|
| 25% | 156 | 0.449 | 0.026 |
| 50% | 230 | 0.496 | 0.035 |
| 75% | 345 | 0.551 | 0.090 |
| 100% | 544 | 0.526 | 0.023 |

AUC dropped from 75% to 100% (slope = -0.026). The probe is **not data-limited** — more training data does not improve it. The gain target is too sparse and noisy for the probe to learn from.

### Accent classifier

An AccentClassifier with identical trunk achieved 50.3% accuracy (macro-F1 = 0.50) on 6-class accent classification, confirming WavLM features encode accent information.

### Layer specialization

| Model | Top 3 layers |
|-------|-------------|
| Gain probe | 0, 24, 2 (edges) |
| Accent classifier | 12, 11, 13 (middle) |

Layer weight correlation: r = 0.56 (p = 0.004) — significantly correlated but not identical. The gain probe uses edge layers (embedding + final), while the accent classifier uses mid-range layers. This divergence suggests the gain probe is learning something other than accent identity, but that something is too weak to be actionable.

## Extension 3: Hallucination Baselines + Confidence Autopsy (EXP-06)

**Question:** Do cheap hallucination detectors explain the routing signal? Why does confidence work?

### Hallucination triggers

| Trigger | Area vs Random | CI |
|---------|---------------|----|
| Confidence (avg_logprob) | 0.0147 | [0.003, 0.028] |
| Hallucination union | 0.0075 | [-0.005, 0.020] |
| No-speech probability | 0.0069 | [-0.005, 0.020] |
| Champion scalar | 0.0058 | [-0.008, 0.020] |
| Compression ratio | -0.0053 | [-0.018, 0.006] |

Compression ratio (text repetitiveness via zlib) is **worse than random**. No-speech probability provides modest signal. The hallucination union does **not** subsume the champion scalar's gain (paired CI includes zero).

### Confidence autopsy

- 56% of hallucinated utterances have avg_logprob **above** the median — **Whisper hallucinates confidently**
- Mean logprob: hallucinated = -0.49, non-hallucinated = -0.45 (small difference)
- Confidence's advantage is **not** primarily explained by hallucination detection — it captures genuine difficulty information

### Indian English slice

- Indian English has the largest mean capped gain (0.123) — routing genuinely helps
- US English has slightly negative mean gain (-0.017) — the careful model is sometimes worse
- Hallucination rates are similar (Indian 3/35, US 3/27), so hallucination doesn't explain the Indian/US gap

**Verdict:** Hallucination flags are not the explanation for confidence's superiority. Confidence captures difficulty information that learned probes have not matched.

## Extension 5: Multi-Task Probe (EXP-08)

**Question:** Does accent supervision regularize the regression probe? Does it degenerate into argmax-in-disguise?

Lambda sweep: loss = (1-λ) * HuberLoss(regression) + λ * CrossEntropyLoss(accent).

| Lambda | Area vs Random | CI | MI(score, accent) | MI(score, speaker) |
|--------|---------------|-----|-------------------|-------------------|
| 0.0 | 0.0047 | [-0.006, 0.017] | 0.244 | 0.306 |
| **0.1** | **0.0108** | **[0.003, 0.022]** | 0.248 | 0.332 |
| 0.3 | 0.0044 | [-0.006, 0.016] | 0.306 | 0.398 |
| 1.0 | -0.0048 | [-0.015, 0.005] | 0.171 | 0.244 |

Lambda = 0.1 is the best learned trigger in Round 2 — the only one besides confidence with a CI excluding zero. At lambda = 0.3, MI(score, accent) rises and routing degrades — the model starts acting as an accent classifier in disguise. At lambda = 1.0, it becomes pure accent classification and is worse than random for routing.

**Paired bootstrap (lambda=0.1 vs champion gain probe):** Area difference = +0.005, CI includes zero. Not significant with 179 test utterances.

## Extension 4: Stats Pooling (EXP-07)

Infrastructure was built (extract_stats method, training scripts, eval pipeline) but not run. Running requires WavLM re-extraction (~1 hour CPU), which was deferred pending Gate 4 approval. This remains a viable follow-up: concatenating mean and standard deviation per WavLM layer (25 × 2048 features) could capture whether difficulty is temporally uniform or concentrated in bursts.

## Synthesis

### Final leaderboard (unique triggers, ranked by area vs random)

| Rank | Trigger | Area vs Random | Significant? | Source |
|------|---------|---------------|-------------|--------|
| 1 | Oracle | 0.0311 | Yes | Upper bound |
| 2 | **Confidence** | **0.0147** | **Yes** | Whisper avg_logprob |
| 3 | Multitask (λ=0.1) | 0.0108 | Yes | Learned |
| 4 | Argmax accent | 0.0103 | No | Group-level |
| 5 | Hallucination union | 0.0075 | No | Heuristic |
| 6 | No-speech prob | 0.0069 | No | Whisper metadata |
| 7 | Probe (gain target) | 0.0058 | No | Learned |
| 8 | Probe (capped WER) | 0.0055 | No | Learned |
| 9 | Multitask (λ=0.0) | 0.0047 | No | Learned |
| 10 | Scalar probe (pilot) | 0.0035 | No | Learned |
| — | Random | 0.0000 | — | Baseline |

### What we learned

1. **The pilot was measuring hallucination, not difficulty.** Uncapped WER contaminated both training and evaluation. Capping at 1.0 is a necessary correction that reverses the pilot's conclusions. The pilot's Pearson r ~ 0.21 was primarily a hallucination-prediction signal.

2. **Confidence is a strong baseline, not a weak one.** The pilot's finding that confidence was "worse than random" was entirely an artifact. Under capped WER, avg_logprob is the second-best trigger — a free feature requiring no training.

3. **The target was necessary but not sufficient.** Training on escalation gain or capped WER fixes the contamination but does not produce a trigger that beats confidence. The gain signal is too sparse (median = 0 for every accent) to learn from with 544 utterances.

4. **The probe is not data-limited for this target.** The learning curve plateaus by 75% of training data. More of the same data won't help — the problem is signal sparsity, not sample size.

5. **A small amount of accent supervision helps.** The multi-task probe at lambda=0.1 is the only learned model achieving significance (area = 0.0108). But the improvement over confidence is not itself significant.

6. **Whisper hallucinates confidently.** 56% of hallucinated utterances have above-median avg_logprob. Hallucination detection is not what makes confidence work — it captures difficulty through a different mechanism.

7. **Routing headroom is modest.** Even the oracle achieves only area = 0.031. With median gain = 0 for every accent, the vast majority of utterances don't benefit from escalation. The small-vs-large model gap in this dataset may simply be too narrow for learned routing to meaningfully outperform simpler strategies.

### Three-way diagnosis resolution

| Bottleneck | Verdict | Evidence |
|-----------|---------|----------|
| Target | **Confirmed and fixed** | Capping WER reverses pilot conclusions; gain probe unstable but capped-WER probe trains well |
| Data | **Not the binding constraint** | Learning curve plateaus at 75%; more data doesn't help for this target |
| Architecture | **Untested** (stats pooling not run) | Mean pooling loses temporal variance; could matter for hallucination-adjacent cases |

The binding constraint is **signal sparsity in the target**, not data volume or architecture. The escalation gain is zero for most utterances, and the non-zero gains are concentrated in 2-3 accents — a pattern that argmax accent captures for free.

### Recommendations for future work

1. **Use confidence as the production baseline.** It requires no training, no feature extraction, and achieves the best non-oracle routing. Any learned model must beat this bar.

2. **Increase the model gap.** The small-vs-large WER difference is too narrow for most utterances. A larger gap (e.g., tiny → large, or distilled → full) would increase the density of routing-beneficial utterances.

3. **Run Extension 4 (stats pooling).** This is the one architectural change that directly addresses temporal information loss. If hallucination triggers are temporally localized (a pause, a burst of noise), std-pooled features could detect them.

4. **Scale the dataset.** While the gain probe is not data-limited at 544 utterances with the current target, a 10x larger dataset with a wider model gap could change the picture.

5. **Consider utterance-level features beyond WavLM.** Duration, silence ratio, speaking rate, and spectral properties could complement the representation — these are cheap to extract and might correlate with difficulty.

## Reproduction

```bash
git clone <repo-url> && cd accentedness-scalar-routing
make setup

# Phase 0: re-score pilot with capped WER
make rescore
make diagnose

# Extensions (all use cached data except ext4)
make ext1          # gain-target + capped-WER probes
make ext2          # learning curve + accent classifier
make ext3          # hallucination baselines + confidence autopsy
# make features-stats && make ext4  # stats pooling (requires re-extraction)
make ext5          # multi-task lambda sweep

# Final comparison
make compare
```

Results are in `experiments/`. The master overlay figure is at `experiments/figures/operating_curves_all.png`. The living leaderboard is at `experiments/COMPARISON.md`.

## AI Use

This project was implemented with assistance from Claude Code (Anthropic). All experimental design, analysis, and interpretation were guided by a pre-approved research plan. AI assistance was used for code generation, debugging, and drafting. See `docs/ai-use-log.md` for details.
