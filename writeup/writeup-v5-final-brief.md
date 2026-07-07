# Accentedness-Scalar Routing for English ASR — Final Brief

**Author:** John Alderete · **Repo:** `accentedness-scalar-routing` (`github.com/aldo-git-bit`, private) · **For:** Deepgram Staff/Research Data Science exercise

---

## 1. Problem definition

Per English utterance, decide — from an acoustic/SSL signal, never the transcript — whether to escalate from a fast default recognizer to a careful one. Success is net WER reduction against escalation cost (a quality-vs-cost curve), not accent-classification accuracy.

---

## 2. Basic approach

The routing decision is the product; detection is only an instrument for making it. The **baseline** is the obvious thing a competent engineer tries first: no-routing floors (default-always / careful-always), a naïve **discrete argmax-accent** router (CommonAccent XLSR), and **free confidence** (Whisper's own `avg_logprob`). The proposed **improvement** is a **learned continuous accentedness/difficulty scalar** — a small probe on WavLM-large hidden states, thresholded to route. Because accent is a *continuum* rather than a class, a scalar that measures *degree of difficulty* fits the phenomenon, needs no accent taxonomy, and doubles as a recognizer-quality (drift) signal. Every trigger — baseline and improvement — is reduced to the same object, a score per utterance swept into an operating curve, so they are compared on identical terms.

---

## 3. Assumptions supporting the approach

- **Detection is the instrument; performance is the metric.** An accent label has no intrinsic value; only the downstream net-WER effect of acting on it counts. A router that improves accuracy but not net WER has failed.
- **Accent is acoustic and dies at the text bottleneck.** Segmental/prosodic cues that define accent are discarded by the transcript, so the trigger must be read from SSL/encoder states, never decoded text.
- **Core problem 1 — the gradient signal (primary).** Accent is gradient and feature-specific (native vowels can co-occur with L2 prosody); a continuous scalar matches this where a hard argmax cannot express "somewhat accented," which is exactly the delicate routing region.
- **Core problem 2 — the unbounded taxonomy (supporting).** Deployments field more accents than any fixed label set enumerates; a label-free scalar absorbs unseen accents by design rather than falling off the end of a taxonomy.
- **Routing is selective spend.** Net WER is only meaningful against a cost axis (escalation rate + latency); always-escalate is just "use the bigger model" and destroys the rationale. The honest deliverable is a curve, not a point.
- **Accent is calibration mismatch, not speaker deficit; labels are sensitive.** Target the calibration gap on under-served varieties; constrain use to performance, not profiling.

---

## 4. Setup

**Data.** EdAcc (conversational, accent-labelled English; CC-BY-SA), a fixed subset of **6 accents × 150 = 900 utterances** (Indian, US, Southern British, Scottish, Irish, Nigerian), resampled 32→16 kHz. Speaker-disjoint **544/177/179** train/val/test folds; the **179-utterance test fold is frozen across all rounds**.

**Models.** Whisper ladder via `mlx-whisper` — `tiny` (39M), `base` (74M), `small` (244M), `large-v3-turbo` (809M), `large-v3` (1.55B); `small → large-v3` is the primary default→careful pair (the ~6× parameter jump is the compute cost routing is spending). Features from `microsoft/wavlm-large` (~316M params; 25 layers × 1024). Discrete baseline from `Jzuluaga/accent-id-commonaccent_xlsr`. Pretrained-first: only the small probe and thresholds train.

**Architecture (the learned scalar probe).** This describes only the trained trigger under test — the other triggers (oracle, random, confidence, argmax-accent) require no such model. Per-layer mean-pooled WavLM states → **learnable weighted sum over layers** (SUPERB-style) → `Linear(1024→256)→ReLU→Dropout(0.1)→Linear(256→1)` (~263K params). **Huber loss (δ=0.1)** for outlier-robust regression; speaker-held-out; percentile-calibrated scores.

**Metrics & target.** All WER **capped at 1.0** before use (neutralizes Whisper hallucination). Target = **escalation gain** = `cap(WER_default) − cap(WER_careful)` (what escalation *buys*, sign kept). Headline = the **operating curve** (net WER vs escalation rate, misrouting cost included), summarized by **area over random** (the total WER a trigger saves vs random escalation across all budgets; random = 0, oracle = ceiling). Ranking quality via **AUC/AP** for `gain > τ`; all claims carry **bootstrap and paired-bootstrap 95% CIs**. (WER is a fraction — net WER ≈0.35 means ~35 errors per 100 words, not 0.35%; values are per-utterance means of capped WER on a hard conversational subset, so they sit above EdAcc's oft-quoted corpus average.)

**Baselines & triggers (every trigger is a score per utterance, thresholded to route).** **Oracle** — routes by *true* gain (escalate the highest-gain utterances first); the upper bound. **Random** — escalates a random fraction; the 0-reference floor. **Confidence** — routes on the default model's `avg_logprob` (low confidence → escalate); free, no training; the bar to beat. **Argmax-accent** — the discrete-taxonomy baseline: escalate whole accent groups by group-level gain. **Champion (learned)** — the best trained scalar of the round (the multitask λ=0.1 probe), carried forward as the improvement under test.

---

## 5. Main findings

**Headroom, not trigger cleverness, is the binding constraint — and it is manufacturable.** Widening the model gap grows the routing prize (oracle), collapses free confidence's *share* of it, and only at the widest gap lets a learned trigger cross confidence:

| Pairing | WER gap | Oracle | Confidence | Learned (champion) | Signal density (gain>0.05) | Confidence share |
|---|---|---|---|---|---|---|
| small → large-v3 | 0.043 | 0.031 | **0.015** | 0.008 | 47 / 179 | 47% |
| base → large-v3 | 0.120 | 0.049 | **0.018** | 0.004 | 84 / 179 | 37% |
| tiny → large-v3 | 0.161 | 0.058 | 0.009 | **0.014** | 98 / 179 | 16% |
| tiny → turbo | 0.165 | 0.061 | 0.005 | −0.001 | 94 / 179 | 8% |

*(area over random; larger = better. `turbo` ≈ `large-v3` headroom at each tier → a cheaper careful path.)*

Supporting results, all on the frozen fold under capped WER:

- **Free confidence is a strong routing scalar — often the best one.** Whisper's own `avg_logprob`, used directly with no training or feature extraction, is the best non-oracle trigger at narrow-to-moderate gaps and **beats the learned scalar at every pairing except the widest** (0.015 vs 0.008 at small→large-v3; 0.018 vs 0.004 at base→large-v3; it only loses at tiny→large-v3, 0.009 vs 0.014). This is the practical headline: **a free scalar is the bar to beat, and the trained probe rarely clears it.**
- **The pilot measured hallucination, not difficulty.** Uncapped WER let repetition-loop blow-ups (Indian-English mean 8.96) dominate; capping *reverses* the pilot — the pilot probe drops **below random**, and confidence, previously reported "worse than random," becomes the **best non-oracle trigger** (0.0147 of oracle 0.0311 ≈ **47%**, free, no training).
- **The signal is genuinely sparse.** Median capped gain is **0 for every accent**; only 47/179 utterances have gain > 0.05, concentrated in **Indian English (mean 0.123)** with US English slightly negative. The residual any trigger can win over confidence (~0.016) is **smaller than the paired-CI width (~±0.012)** at n=179 — the narrow setup is underpowered to separate a good learned trigger from free confidence. The probe is **not data-limited** (learning curve plateaus by 75%); the ceiling is target sparsity.
- **Gradient thesis (problem 1):** a *little* accent supervision helps — the multitask λ=0.1 probe (0.0108) is the best learned trigger — but pushing the scalar toward **discrete accent identity** (higher λ) *degrades* routing as its mutual information with accent rises. **Degree beats identity.**
- **Taxonomy thesis (problem 2):** the discrete argmax router (0.0103) is respectable **only because the 6-accent set is closed** and the gain concentrates in one group — the exact advantage that would not survive an unseen accent. Label-free triggers (confidence, scalar) match it **without enumerating any taxonomy.**
- **Methodological caution:** composite and temporal-std combiners raise classification **AUC** (to 0.78, +11 pts) while *losing* on area-over-random — **high AUC ≠ better routing.**

---

## 6. Discussion

**Detection-as-instrument, vindicated.** Insisting on the honest metric — net WER against a cost axis, misrouting folded in — is what exposed that the pilot's apparent win was hallucination contamination and that the bar to beat is a *free* signal. An accuracy-first framing would have shipped a worse-than-random router that looked good.

**The gradient thesis is conditionally confirmed.** The correct routing object is a continuous quality scalar: confidence proves the free version works, and the multitask sweep proves *degree* routes better than *identity*. But the **learned** scalar's marginal value over free confidence only materializes once the model gap is wide — it earns its keep in the wide-gap regime, not the narrow one. Practically: **use free confidence until the gap is wide; then a learned or accent-adapted trigger can pay off** (and prefer the cheaper `turbo` careful path; optimize area, not AUC).

**The taxonomy thesis is confirmed in principle but untested at its limit.** Label-free matches discrete without enumerating, and forcing a taxonomy into the trigger hurts — but a closed 6-accent set flatters the discrete baseline. The decisive experiment (below) is open-set.

**Limitations / failure modes.** n=179 is power-limited (several key CIs include zero, including the wide-gap crossing); results are single-corpus and closed-set; the scalar leans **speaker** over accent in MI (defensible under speaker-held-out splits, but bounds unseen-speaker generalization); and `large-v3` does not fully rescue conversational Indian/Nigerian English, capping the prize itself.

**Next steps (ranked).** (1) **Accent-adapted careful path** on the Indian-English slice — "try *differently*, not just harder" — gated by a verify-before-trust WER check; the strongest on-thesis test of accent-*aware* routing. (2) **Open-set / unseen-accent evaluation** (e.g. AfriSpeech-200) — the decisive stress test of the taxonomy thesis, where a discrete classifier falls off its labels and a scalar should degrade gracefully. (3) **Scale test power** at a wide gap (scoped Common Voice + wide gap) to resolve the crossing. (4) A **denoised, decision-aligned target** (separate hallucination-vs-not from graded difficulty). (5) **Escalation-specific pooling** aimed at confidence's confidently-hallucinated blind spot. (6) Operationalize the **drift detector** — the same scalar distribution that routes also monitors.

**AI-tool use.** Building, caching, plotting, and drafting were delegated to Claude Code under **pre-registered decision rules** (each experiment's outcome, including nulls, mapped in advance to a conclusion). Key corrections were human-owned: the contaminated uncapped-WER metric (which inverted the pilot's story) was caught and fixed, and **AUC was rejected as a routing proxy** once it diverged from area-over-random. Everything was validated on a frozen test fold through one shared scorer with bootstrap CIs, and every headline claim was required to survive a **paired bootstrap against free confidence**, not merely look good alone.

**Reproduction.** `make setup && make reproduce` (pilot) / `make reproduce-v2` / `make reproduce-v3` run each round end-to-end from cache; full instructions and the `experiments/COMPARISON.md` leaderboard are in the README.
