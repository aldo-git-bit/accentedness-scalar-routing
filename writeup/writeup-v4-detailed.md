# Accentedness-Scalar Routing for English ASR
### A Detailed Technical Report on Three Rounds of Experimentation

**Author:** John Alderete
**Repository:** `accentedness-scalar-routing` (`github.com/aldo-git-bit`, private)
**Context:** Take-home exercise for a Staff / Research Data Science role at Deepgram
**Document:** `writeup-v4-detailed.md` — consolidated across the pilot, Round 2, and Round 3

---

## 1. Abstract

This project asks a narrow, product-shaped question inside the broad topic of accent handling in speech recognition: **given a fast default recognizer and a slower careful one, can a signal read from the audio decide, per utterance, when escalation is worth it — and does a *learned, continuous accentedness scalar* do this better than free or discrete alternatives?**

The work is deliberately framed so that detection is an *instrument* and the routing *behavior* is the product. Success is therefore never classification accuracy; it is a quality-versus-cost operating curve — net word error rate (WER) against escalation rate — evaluated on real, conversational, accent-labelled speech (EdAcc), with the cost of wrong routing folded in.

Across three rounds and twelve experiment folders, the central empirical finding is not about any one trigger. It is that the **binding constraint is headroom**, not trigger cleverness:

- On the original model pair (Whisper-small → large-v3), the entire routing prize — the oracle's area over a random router — is only **0.031**. A *free* signal, Whisper's own `avg_logprob` confidence, captures **~47%** of it with no training. The residual any learned trigger can compete for (~0.016) is smaller than the paired-bootstrap confidence-interval width (~±0.012) at n = 179 test utterances. The setup is **underpowered to distinguish a good learned trigger from free confidence.**
- Headroom is **manufacturable**: widening the model gap (tiny → large-v3) nearly doubles the oracle prize (0.031 → 0.058) and more than doubles the number of routing-relevant utterances. Only at the widest gap does a learned trigger *cross* confidence for the first time (0.014 vs 0.009) — but not yet significantly on 179 utterances.
- The pilot's headline numbers were **contaminated by hallucination** (uncapped WER > 1.0 from Whisper repetition loops). Capping WER at 1.0 *reverses* the pilot's conclusions: the pilot probe drops below random, and confidence — which the pilot reported as "worse than random" — becomes the strongest non-oracle trigger.

Read against the two core problems the scalar was chosen to address, the results are a **conditional resolution**: the *gradient* framing is vindicated (a little accent supervision helps; forcing the signal toward discrete accent identity *hurts*; the useful trigger is a continuous difficulty scalar), and the *unbounded-taxonomy* framing is vindicated in principle but not stress-tested here (the discrete argmax-accent baseline is only strong because the 6-accent set is closed and the gain concentrates in one or two of them — exactly the advantage that would not survive open-set deployment). The honest production takeaway: **use free confidence until the model gap is wide; reach for a learned or accent-adapted trigger only where headroom demonstrably exists.**

---

## 2. Problem Statement

### 2.1 The assignment, reframed

The assignment asks us to "detect whether a speaker belongs to a target accent category, and define a practical way to use that signal in a speech product." Taken literally this points at a classifier. The reframing that drives this entire project — argued in full in the knowledge-base docs and summarized here — is that **the accent label has little intrinsic value; the value is the change produced in a downstream outcome by acting on it.** Detection is an instrument; the product is the routing decision.

Concretely, the object of study is a **per-utterance routing decision**: for each utterance, choose a **default path** (a fast, cheap recognizer) or a **careful path** (a larger, slower, or adapted recognizer), based on a signal read **from the audio or its self-supervised representation, never from the decoded transcript**. The "improvement" from routing is that hard utterances are escalated while the easy majority stay cheap.

This reframing is not stylistic. It changes what counts as success, what the baselines are, and — most importantly — what failure looks like. A router that improves classification accuracy but does not move net WER has failed. A router that helps a common accent while silently degrading a rarer one has failed. A router that "improves WER" by escalating most of its traffic has also failed, because it has merely rediscovered "use the bigger model" and destroyed the rationale for routing, which is **selective spend**.

### 2.2 Core problem 1 — the gradient accentedness signal (primary)

Accent is not a discrete label. It is a **gradient, feature-specific** property: a speaker may carry native vowels with second-language prosody, or shade continuously from lightly to heavily accented. A discrete argmax classifier forces hard categories onto a continuous phenomenon and cannot express "somewhat accented" — which is precisely the region where routing decisions are most delicate.

Two consequences follow, and together they are the primary motivation for the design:

1. **A continuous scalar matches the phenomenon.** Modelling *degree* of accent/difficulty, rather than identity, is the representation that fits a gradient cause.
2. **A difficulty scalar doubles as a quality signal.** Because it measures *how hard / how accented* rather than *which accent*, the same scalar is a general recognizer-quality signal, and therefore a natural **drift / monitoring** signal — the routing trigger and the monitoring trigger are the same object.

Additionally, ASR degradation on accented speech is not a speaker deficiency; it is **model calibration mismatch** — the recognizer is under-fitted to varieties under-represented in its training data. The product target is therefore "close the calibration gap on under-served varieties," and any product use is constrained to *performance improvement, not profiling.*

### 2.3 Core problem 2 — the unbounded taxonomy problem (supporting)

Discrete accent inventories break under distribution shift. Real deployments field far more accents than any fixed label set can enumerate (a call centre may hear 1,000+ first languages), and N-way classification degrades sharply as N grows. A scalar's **label-free** nature absorbs unseen accents as a side benefit rather than by design: it never enumerates a taxonomy, so it cannot fall off the end of one.

This is a *supporting* argument, not the core one. The primary case for the scalar is gradience and the instrument framing above; the taxonomy argument is convergent evidence that pushes any "accent-aware" approach away from discrete classify-then-route and toward scalar/similarity/quality triggers.

### 2.4 Routing as selective spend — the cost axis

Routing exists to spend selectively. Net WER is only meaningful **against a cost axis**; without one it is trivially maximised by always escalating. The cost axis has two currencies:

- **Quality cost of misrouting** — an utterance sent down the wrong path (a non-target degraded by the careful model, or a target that missed its improvement). This lives *inside* the net-WER metric, scored against an oracle router.
- **Operational cost** — latency and compute/$, paid on *every* escalation, right or wrong. In this local prototype there is no API bill, so **escalation rate** (fraction sent to the careful path) and **added latency** stand in for the dollars that are literal in production.

The honest deliverable is therefore a **trade-off, not a point**: a quality-vs-cost operating curve, with the operating point chosen where the marginal WER gain justifies the marginal cost.

### 2.5 What counts as success and failure

- **Primary objective:** net WER (capped; misrouting cost folded in) of the routed system vs. the no-routing floor, **measured against escalation rate** — reported as an operating curve, summarized by net WER at fixed escalation budgets and by *area over the random router.*
- **Success** = a net-WER reduction that survives misrouting cost at an acceptable escalation rate and latency, ideally concentrated where the baseline is weakest (accented / low-confidence speech), with **no silent per-slice regression.**
- **Failure** = improved classification with no net-WER movement; a per-slice regression hidden by an aggregate gain; or a gain erased by misrouting or latency cost.
- **Valid results include conditional and negative findings.** "Routing helps Indian English but not US English" is a first-class result, not a disappointment — honest per-slice reporting is itself evidence of good evaluation.

### 2.6 Assumptions (stated explicitly)

- **English**, cascaded / text-emitting ASR (so accent survives only in audio/encoder/metadata, not transcript).
- Evaluation on **real accented speech with references**; self-reported accent labels treated as **noisy proxies**, standardized labels as the working label.
- Accent framed as **calibration mismatch, not speaker deficit.**
- **Pretrained-first:** the only training is the small probe (plus learnable layer weights) and threshold calibration. No ASR fine-tuning, no training from scratch.
- The trigger is read from **acoustics / SSL / metadata**, never from decoded text.

---

## 3. Context That Supports the Problem

This section compresses the reasoning developed in the knowledge base (`what-is-accent.md`, `accent-routing-sota.md`, `accent-routing-scoping.md`) into the specific claims that shape the build. It exists to justify *why* the problem is posed the way §2 poses it.

### 3.1 What accent is — and why the trigger must be acoustic, not textual

Accent is a **probabilistic configuration of many weak cues** — vowel-space shifts, VOT, rhoticity, allophony, prosody, voice quality — not a single decisive feature. The engineering-decisive fact is a single column in the decomposition: **which of these cues survive the ASR text bottleneck.** Almost none do. Narrow accent lives in segmental and prosodic acoustics that a transcript has already discarded; only *dialect* markers (lexicon, morphosyntax — *hoagie*, *y'all*, *might could*) survive into text. Phonemic mergers and phonotactic repairs survive only indirectly, as *errors*.

The consequence is a hard design constraint: **compute the routing signal from SSL encoder hidden states (WavLM / wav2vec2 / HuBERT), never from decoded text.** A cascade that classifies accent or estimates difficulty from the transcript has already thrown away the evidence. This is why the trigger is a probe on WavLM features and not, say, an LLM reading the hypothesis.

A second decision-relevant fact: accent content is **layer-distributed** in SSL models (phonetic/accent information peaks in mid-to-upper layers; prosody sits in the middle third), which motivates a **learnable weighted sum across layers** rather than hard-picking one layer, and **mean-pooling over the utterance** because accent is distributed across the whole utterance rather than localized. A known confound — SSL features leak **speaker identity**, and accent classifiers frequently learn the speaker instead of the accent — dictates **speaker-held-out** evaluation as mandatory, and motivates the mutual-information leakage checks reported later.

### 3.2 The landscape — where routing lives

Accent handling is usually reduced to one axis (accent-agnostic vs accent-aware), but that hides the decision that matters for a product: **whether you branch computation at inference, and on what signal.** Separating those gives a 2×2. The historically dense cells are the *single-model* column (invariance training; accent conditioning). The *branch-at-inference* column — where routing lives — is thin and recent, and one of its cells (quality-driven escalation as an *accent* strategy) is nearly empty in the literature, borrowed from generic confidence-estimation and model-cascade work. That emptiness is the opportunity this project occupies.

Two further landscape facts shape the metric and the substrate:

- **The literature reports the specialist's WER, rarely the *net* WER of a routed system including misrouting cost and latency.** Net-of-misrouting is the honest metric and it is usually missing — so reporting it *is* the demonstration of judgment.
- Everything is **cascade-era**: the methods assume a text-emitting ASR stage to route around. In an audio-native / speech-to-speech model the "route on acoustics not text" constraint dissolves and routing becomes internal — out of scope here, but the right framing for "what's next."

### 3.3 Why a scalar — the candidate-selection logic

The scoping doc defines a design space (task definition, baseline, improvement, flywheel) and deliberately does not commit to one instantiation. Six concrete proposals were generated against that space and evaluated for feasibility, novelty, and fit. The **accentedness scalar** was selected because it uniquely satisfies four criteria at once:

1. **Feasible** in a ~5-hour, pretrained-first, single-Mac budget (only a small probe trains).
2. **Novel-ish** — it sits in the sparse right-hand column and treats accent as a *degree* feeding a quality-escalation policy, the near-empty cell.
3. **Addresses both core problems** — it matches the gradient nature of accent (problem 1) and is label-free, so it sidesteps the unbounded taxonomy (problem 2).
4. **Doubles as a drift detector** — the same scalar distribution that routes also monitors, which is strongly on-brand for a "data factory" role.

The discrete argmax-accent classifier (CommonAccent XLSR) is retained not as the contribution but as **the key baseline the scalar must beat** — it is the obvious, taxonomy-bound thing a competent engineer would try first, and beating it is how the gradient/label-free thesis earns its keep.

---

## 4. Setup (Methods, Generalized Across All Experiments)

The three rounds used different *metrics of record* — the pilot reported uncapped WER and Pearson correlation; Round 2 introduced capped WER, AUC/AP, and bootstrap CIs; Round 3 added a headroom parameterization across model pairs. This section describes the union of that machinery once, so the experiment-by-experiment accounts in §5–§6 can refer back to a single vocabulary. **Every term the later sections rely on is defined here explicitly**, including the two that most often cause confusion: the change in the probe *target* between rounds, and the definition of *area over random*.

### 4.1 Data

**Primary corpus — EdAcc (Edinburgh International Accents of English).** Conversational English dyads, self-reported accents across many first languages, per-speaker linguistic profiles, CC-BY-SA. It is purpose-built as a bias benchmark and is the most product-realistic evaluation available: the best 680k-hour models degrade sharply on its accented conversational speech. EdAcc ships **validation + test only (no train split)** and at **32 kHz**, which must be resampled to **16 kHz** for both WavLM and Whisper.

**Subset used throughout.** A fixed subset of **6 accents × 150 utterances = 900 utterances**: Indian, US, Southern British, Scottish, Irish, and Nigerian English. This subset is held constant across all rounds so that every experiment is comparable.

**Splits.** The validation+test pool is re-partitioned into **speaker-disjoint** probe-train / probe-val / test folds of **544 / 177 / 179** utterances, accent-stratified where feasible, fixed seed. Speaker-disjointness is non-negotiable because SSL features leak speaker identity; without it the probe could score well by memorizing speakers. **The 179-utterance test fold is frozen across all three rounds** — every trigger, old and new, is scored on the identical speakers, so cross-experiment comparisons are legitimate.

**Read-speech contrast (deferred).** Common Voice via Mozilla Data Collective was scoped as an optional read-vs-conversational contrast and as a data-scaling lever, but deferred: Round 2's learning curve showed the current target is not data-limited, so paying the domain-shift tax was not justified until a wider model gap exists.

### 4.2 Models

- **ASR ladder (all via `mlx-whisper` on Apple Silicon):** `tiny`, `base`, `small`, `large-v3-turbo`, `large-v3`. Whisper-small is the pilot/Round-2 **default** path; large-v3 is the **careful** path. Round 3 promotes the full ladder to sweep the default∈{tiny,base,small} × careful∈{turbo,large-v3} grid. For each utterance × model the cache stores the hypothesis, capped WER, `avg_logprob`, `no_speech_prob`, and wall-clock.
- **SSL feature extractor:** `microsoft/wavlm-large` (25 hidden-state layers × 1024 dims), run on CPU at this scale to avoid MPS op-coverage flakiness. A NaN-guard smoke test asserts finite hidden states (a known loading bug is avoided by loading without `device_map=`). WavLM features are **default-agnostic**, so the same feature cache is reused across every model pairing.
- **Discrete-accent baseline:** `Jzuluaga/accent-id-commonaccent_xlsr-en-english` — the off-the-shelf classifier whose signal drives the argmax-accent router.
- **Accent-adapted careful path (deferred, gated):** `Tejveer12/Indian-Accent-English-Whisper-Finetuned`, a large-v3-turbo finetune, held behind a verify-before-trust guard (see §5, EXP-12).

### 4.3 System architecture

The routed system is a **shared front end fanning out to a trigger and to ASR**: WavLM features are computed once and used both by the learned probe and (implicitly) by the analysis, while Whisper decodes provide the metadata triggers (confidence, no-speech). The routing decision is: score each utterance with a trigger, compare to a threshold, send above-threshold utterances to the careful model and the rest to the default. Sweeping the threshold traces the operating curve. This shared-front-end shape is intentional — it is cheap and it mirrors a production "data factory" where one representation feeds many consumers.

### 4.4 The learned probe (the contribution)

A lightweight probe (~**263K parameters**) on WavLM-large representations:

- **Input:** per-layer mean-pooled hidden states, 25 layers × 1024 dims.
- **Architecture:** `LearnableWeightedSum(25 layers)` → `Linear(1024 → 256)` → `ReLU` → `Dropout(0.1)` → `Linear(256 → 1)`.
- **Layer weighting:** the softmax-weighted sum across layers is learned (SUPERB-style), initialized uniform. The learned weights are reported per experiment as a diagnostic of *which depth carries the signal.*
- **Pooling:** mean over time to one vector per layer per utterance (Round 3 adds a std-pooled variant — see EXP-11).
- **Loss — Huber (δ = 0.1).** The probe is a regressor, so it needs a loss that measures how far each predicted scalar is from its target. Huber loss is a **hybrid of squared error (L2/MSE) and absolute error (L1/MAE)** that switches between them based on the size of the residual `r = prediction − target`:

  ```
  L(r) = ½·r²                    if |r| ≤ δ      (quadratic, like MSE)
  L(r) = δ·(|r| − ½·δ)           if |r| >  δ     (linear,    like MAE)
  ```

  For **small** errors it behaves like MSE — smooth and continuously differentiable near zero, so training converges cleanly. For **large** errors it behaves like MAE — the penalty grows only *linearly*, not quadratically. The practical consequence is in the *gradient*: MSE's gradient is `2r`, which keeps growing with the error, so a handful of huge residuals dominate the update and drag the fit toward outliers. Huber's gradient **saturates at ±δ** once `|r| > δ`, capping any single outlier's influence at a constant. That makes it **robust regression** — a good fit to the bulk of the data without being hijacked by extreme points. Here `δ = 0.1` sets that transition low, because the target has a heavy tail (hallucination-inflated WER in the pilot; sparse, spiky escalation gain from Round 2 on) that we do not want the probe chasing. **Important caveat, established empirically (§6.3.1):** in the pilot, Huber bounded each outlier's *gradient* but could not save a *contaminated target* — the loss was still asked to rank-order a hallucination tail, so the fix that actually mattered was **capping the WER target itself** at 1.0, not the choice of loss.
- **Training:** speaker-disjoint folds; AdamW at lr = 1e-3; early stopping on validation loss (patience 10).
- **Score calibration:** percentile normalization (2nd/98th percentile of training predictions), clipped to [0, 1], so the scalar is threshold-swept on a common scale.

The **target** the probe regresses is the single most important variable that changed between rounds; it is defined precisely in §4.5.

### 4.5 Targets — explicit definitions (this changed between rounds)

All WER is computed with a consistent English text normalizer (lowercase, strip punctuation) before scoring.

- **Word error rate (WER):** `(substitutions + insertions + deletions) / reference_words`. It is **unbounded above** — massive insertion (e.g. a repetition loop) can push a single utterance's WER far above 1.0.

- **Capped WER** — `cap_wer(w) = min(w, 1.0)`. Introduced in Round 2 and applied **before all aggregation and before all target construction** from that point on. Capping treats "the model produced garbage" and "the model produced 3× garbage" as equally bad (both = 1.0), which is the correct decision-theoretic stance for routing: you cannot lose more than the whole utterance.

- **Pilot target — default-model WER (uncapped).** The pilot probe regressed Whisper-small's raw per-utterance WER as a difficulty proxy. **This target is contaminated** (see §6.3.1): a heavy hallucination tail (WERs of 223, 74, 55…) dominates the loss, so the probe learned to rank hallucinations rather than accent difficulty. Numbers derived from this target — including the pilot's Pearson r ≈ 0.21 — are **not valid difficulty signals** and are quarantined throughout this report.

- **Round-2+ target — escalation gain (capped).** The decision-theoretically correct target. For each utterance:

  ```
  escalation_gain = cap_wer(wer_default) − cap_wer(wer_careful)   ∈ [−1, 1]
  ```

  Positive gain means the careful model *recovers* errors and escalation is worthwhile; **negative gain means the careful model is worse**, and the sign is kept (escalating a negative-gain utterance is a real cost). This is what a router should predict — *what escalation buys*, not *how bad the default is*. The distinction matters because default-WER and gain diverge exactly on the hardest utterances, where both models fail and escalation buys nothing despite a high default WER.

The **why** of the switch: regressing default WER conflates "hard for everyone" with "helped by escalation." Round 2's Extension 1 (EXP-04) isolates the target as the only changed variable to test whether this conflation was the bottleneck.

### 4.6 Metrics — explicit definitions

**Operating curve (the headline object).** Sweep the trigger's decision threshold. At each threshold: route above-threshold utterances to careful, the rest to default; compute **net WER** = mean over utterances of `cap_wer` of the *chosen* path, and **escalation rate** = fraction routed to careful. Plotting net WER (y) against escalation rate (x) as the threshold sweeps traces a curve from `(0%, default-always WER)` to `(100%, careful-always WER)`. **Misrouting cost is automatic:** escalating a negative-gain utterance raises net WER; failing to escalate a positive-gain one leaves WER unreduced. A good trigger bows the curve *below* the endpoints' chord — it buys more WER reduction per unit of escalation, early.

**Reference routers (bracket every figure).**
- **Default-always / careful-always** — the two curve endpoints.
- **Oracle** — routes strictly by true gain (highest first); the upper bound.
- **Random** — escalates a random fraction; in expectation the straight chord between the endpoints; the lower bound / sanity line.
- **Confidence** — routes on the default model's `avg_logprob` (low confidence → escalate). *Default-specific*: each pairing has its own confidence trigger.
- **No-speech probability** — the default model's `no_speech_prob`; also default-specific.
- **Argmax-accent** — the discrete-taxonomy baseline: route whole accent groups by group-level mean WER/gain.
- **Hallucination flags** — `compression_ratio` (text gzip repetitiveness) and `no_speech_prob`, individually and as a union.
- **Champion probe** — the best learned scalar of the round, carried forward.

**Area over random (`area_vs_random`) — the intuition.** This is the project's single "how good is this trigger, overall?" score, and it exists to solve a specific annoyance: an operating curve is a whole *curve*, and a trigger can look better at a 10% budget but worse at 40%, so comparing triggers means comparing curves, not numbers. Area over random collapses the curve to one number by asking: *across every possible escalation budget at once, how much total WER does this trigger save me versus just escalating at random?* A trigger's job is to spend a limited escalation budget on the utterances that benefit most; a random router spends it blindly; a good trigger front-loads the wins, so its curve dips below the random line early. The size of the gap between the two curves — added up over all budgets — is exactly that "how much smarter than a coin flip" quantity. It's deliberately **budget-agnostic**, so you can rank triggers without first committing to one operating point. In use: **random sits at 0 by construction** (the floor), the **oracle sits at the top** (the ceiling — the best any router could do given the models), and every real trigger lands in between; dividing a trigger's value by the oracle's tells you *what fraction of the achievable prize it captured* (e.g. confidence's 0.0147 / 0.031 ≈ 47%). Higher is better; negative means the trigger is actively worse than random.

**Area over random — defined precisely.** Formally, the single scalar that summarizes a curve is:

```
area_vs_random = ∫₀¹ [ netWER_random(r) − netWER_trigger(r) ] dr
```

integrated over escalation rate r. By construction **random = 0**. A **positive** value means the trigger reduces WER faster than random as budget is spent; **negative** means *worse than random.* The **oracle sets the practical ceiling** (0.0311 on small→large-v3 under capped WER). Because the integrand is in WER units, the metric's scale depends on capping — this is why the pilot's uncapped oracle (0.65) and uncapped "confidence = −1.33" live on a completely different scale from the capped 0.031-scale numbers and **cannot be compared across the cap boundary.** All valid cross-round comparisons use capped `area_vs_random`.

**Net WER at fixed budgets.** `netWER@{10,20,30,50}%` — the curve's height at fixed escalation rates, for reading off performance at a chosen spend.

**Decision scorecard — AUC and AP for `gain > τ`.** Routing is fundamentally a **ranking/threshold** problem, so the cleanest classifier-style scorecard is the area under the ROC curve (AUC) and average precision (AP) for the *binary event* "escalation gain exceeds τ," at τ = 0 and τ = 0.05. This measures whether the trigger **orders** utterances by routing benefit, independent of where a threshold is set.

**Rank correlations.** Spearman (rank) and Pearson (linear) between trigger score and capped gain. **Pearson is nearly uninterpretable on this target** — the gain distribution is a spike at zero plus a heavy tail — which is exactly why the pilot's Pearson-centric evaluation was misleading and Round 2 moved to AUC/AP + curve area.

**WER gap (headroom parameter, Round 3).** Per pairing, `mean(cap_wer_default) − mean(cap_wer_careful)` — how much room the careful model has to help *on average.* The grid sweeps this from 0.043 (small→large-v3) to 0.165 (tiny→turbo).

**Signal density.** The count of utterances with capped gain > 0.05 — how many utterances routing can actually act on. Ranges from 47/179 (narrow gap) to ~98/179 (wide gap).

**Leakage — mutual information.** `MI(score, speaker)` and `MI(score, accent)`, with their ratio, quantify whether the scalar has collapsed into a **speaker recognizer** (the SSL confound) or an **accent recognizer** (argmax-in-disguise). A pure difficulty scalar should be dominated by neither.

### 4.7 Statistical methodology

- **Bootstrap CIs.** 95% intervals from ≥1000 utterance resamples, recomputing the full operating curve per resample, on every summary scalar and as a band on the curve.
- **Paired bootstrap.** Pairwise claims ("trigger A beats B at 20%") use a **paired bootstrap on the difference**, never two overlapping marginal bands. In Round 3 the bar to beat is always **confidence**, so every "does X help" claim is a paired bootstrap *vs confidence*.
- **Power discipline.** At n = 179 the study is power-limited. The rule is **select on validation, confirm on test, report the null honestly** — no threshold-hunting on the test fold. Paired-difference CIs are reported regardless of sign.

### 4.8 Comparability rules (what makes twelve runs one study)

1. **Frozen test fold** — the identical 179-utterance speaker-disjoint fold everywhere; never re-split. Learning-curve subsampling touches the *train* fold only, and subsamples **by speaker.**
2. **One shared scorer** — `eval_common.py` is the single source of curve/metric logic; no experiment forks it. If a metric changes, it changes there and *everything* is re-scored.
3. **Capped WER everywhere**, from Round 2 on; report both **micro** and **macro** (per-accent-averaged) aggregates.
4. **Curves with CIs, not points.**
5. **Reference set carried forward** in every figure — oracle, random, confidence, argmax-accent, and the round's champion — so each new trigger is read against a fixed backdrop.
6. **Determinism** — fixed seeds; key library versions logged into every report.

---

## 5. The Experiments (Complete Ledger)

Twelve experiment folders span the three rounds. Not all are substantive: some are pipeline scaffolding, one is infrastructure that was built but deferred, and one is gated and not run. The **Finding?** column classifies each on the report's significance criterion (§6.1): a finding is *significant* only if it is both **uncorrupted** (not an artifact of the contaminated pilot metric) **and meaningful** (a real, interpretable outcome, positive or negative). A result can be statistically clean yet not meaningful (e.g. a high correlation on a contaminated target), and a result can be a genuine finding while being a *null* (e.g. "no learned trigger beats confidence at narrow gaps").

| EXP | Round | Goal / question | Key result | Finding? |
|-----|-------|-----------------|-----------|----------|
| **01** | Pilot | Build the data → ASR → WER → feature pipeline; train the first scalar probe | Pipeline stands up end-to-end; probe trains (best epoch 4, val loss 0.271) | Infrastructure — not a scientific finding |
| **02** | Pilot | Does the learned scalar route better than baselines? (uncapped) | Reported scalar ≈ argmax, "confidence worse than random" | **Corrupted** — reversed by capping; not valid |
| **03** | Pilot | Flywheel: hard-case mining + drift signal (optional stretch) | Extreme misrouting-cost utterances surfaced and correctly high-scored | Demonstrative — supports the drift/flywheel thesis, not a metric result |
| **04** | R2 | Does regressing **escalation gain** (not default WER) fix the weak signal? | Gain probe 0.0058, capped-WER probe 0.0055, pilot 0.0035; none beats confidence; gain probe unstable | **Significant (null):** target was necessary but *not sufficient* |
| **05** | R2 | Is the probe **data-limited**? Does it specialize like an accent classifier? | Learning curve **plateaus/declines** by 75%; accent classifier hits 50.3%; probe uses edge layers, classifier uses middle | **Significant:** not data-limited; signal is target-sparse, not sample-starved |
| **06** | R2 | Do trivial **hallucination flags** explain the gain? Why is confidence good? | Hallucination union 0.0075 < confidence 0.0147; **56% of hallucinations are *confident*** | **Significant:** confidence is genuine difficulty, not hallucination detection; confidence has a blind spot |
| **07** | R2 | Stats pooling (mean+std over time) | Infrastructure built; **not run** (deferred pending re-extraction gate) | Deferred — realized later as EXP-11 |
| **08** | R2 | Does accent **multi-task** supervision help, or collapse to argmax? | λ=0.1 best learned trigger (0.0108, CI excludes zero); λ≥0.3 degrades as MI-to-accent rises | **Significant:** gradient-over-discrete evidence; a *little* accent helps, forcing identity hurts |
| **09** | R3 | Is **headroom manufacturable**? Does confidence keep winning as the gap widens? | Oracle 0.031→0.058 (×1.9); confidence share 47%→16%; champion **crosses** confidence at tiny→large-v3 (0.014 vs 0.009) | **Significant (headline):** headroom grows; confidence degrades; regime located |
| **10** | R3 | Does a **composite** (confidence + learned + acoustic) beat confidence? | Composite test AUC 0.78 at wide gap but **lower area-vs-random than confidence** in both regimes | **Significant (null + methodological):** the AUC–area disconnect |
| **11** | R3 | Does **temporal std** catch confidence's *confidently-hallucinated* blind spot? | +11 AUC points, but area gain +0.003 (CI includes zero); **no advantage on the blind-spot subset** | **Significant (null):** temporal std captures general difficulty, not escalation-specific difficulty |
| **12** | R3 | Does an **accent-adapted** careful path open headroom a bigger *general* model cannot? | **Deferred** behind a verify-before-trust guard; not run | Not run — the strongest on-thesis next step |

Nine of the twelve produced substantive findings (EXP-02 is corrupted-but-instructive; EXP-01/03 are pipeline/demonstrative; EXP-07/12 are deferred). The scientific spine is **EXP-04 → 05 → 06 → 08** (Round 2's diagnosis) and **EXP-09 → 10 → 11** (Round 3's regime-finding), with EXP-02's corruption being the finding that made the rest necessary.

### 5.1 What each substantive experiment showed (one paragraph each)

**EXP-02 (pilot eval) — the corruption that reframed everything.** As reported under uncapped WER, the scalar tied the argmax baseline and confidence looked "worse than random" (area −1.33). Both claims are artifacts: uncapped WER lets a handful of hallucination blow-ups dominate the integrated area. This is not a valid result, but discovering *that* it was invalid is what launched Round 2.

**EXP-04 (gain target) — target was necessary, not sufficient.** Switching the probe's target from contaminated default-WER to capped escalation-gain (and, as a control, to capped default-WER) improved training stability — the capped-WER probe reached a clean val Pearson of 0.675 — but **did not** produce a router that beats confidence or dominates argmax. The gain-target probe never stabilized (val r oscillated near zero). Verdict: the contaminated target was a real problem and had to be fixed, but fixing it did not unlock a strong trigger.

**EXP-05 (diagnostics) — not data-limited; the target is sparse.** The learning curve for the champion probe *fell* from 75% to 100% of the training speakers (val AUC 0.551 → 0.526), so more of the same data does not help. An accent classifier with the identical trunk reached 50.3% accuracy and — crucially — **specialized its layer weights to the middle layers (12, 11, 13)** while the gain probe leaned on **edge layers (0, 24, 2)**. That divergence (weight correlation r = 0.56) shows the gain probe is learning *something other* than accent identity, but that something is too weak to act on. The binding constraint is **signal sparsity in the target**, not sample size.

**EXP-06 (hallucination autopsy) — confidence is real, and it has a blind spot.** Cheap hallucination flags (compression ratio, no-speech prob, their union) do **not** subsume confidence: the union scores 0.0075 vs confidence's 0.0147, and compression ratio is actually *worse than random.* So confidence's advantage is genuine difficulty information, not hidden hallucination detection. The autopsy also produced Round 3's most actionable clue: **Whisper hallucinates *confidently* — 56% of hallucinated utterances have above-median `avg_logprob`** — so a "low-confidence → escalate" rule systematically *misses* exactly the catastrophic cases, which is a well-defined blind spot to attack.

**EXP-08 (multi-task) — the cleanest gradient-vs-discrete result.** Adding an accent-classification head with weight λ to the regression probe and sweeping λ ∈ {0, 0.1, 0.3, 1.0}: λ=0.1 is the **best learned trigger of Round 2** (area 0.0108, CI excludes zero) — a *little* accent supervision regularizes the difficulty scalar. But at λ=0.3 and λ=1.0 the routing **degrades** as `MI(score, accent)` rises (0.248 → 0.306 → falls only when the model becomes a pure, useless-for-routing classifier). Pushing the scalar toward discrete accent identity makes it *worse* at routing — direct evidence for the gradient framing. (The λ=0.1 edge over the plain gain probe is itself not significant on 179 utterances.)

**EXP-09 (headroom grid) — the headline.** Sweeping six default×careful pairings shows the oracle prize nearly doubling with the WER gap (0.031 → 0.058), confidence's *share* of it collapsing (47% → 16%), and — for the first time in the whole project — a **learned trigger crossing confidence** at the widest gap (champion 0.014 vs confidence 0.009 at tiny→large-v3). Signal density more than doubles (47 → ~98 routing-relevant utterances). And large-v3-turbo delivers **nearly the same headroom as large-v3 at lower cost**, a selective-spend result independent of trigger quality.

**EXP-10 (composite) — the AUC–area disconnect.** A logistic combiner over [confidence, no_speech_prob, champion score, duration, silence ratio, speaking rate] achieves impressive *classification* AUC (0.78 at the wide gap) but **lower area-vs-random than confidence alone** in both regimes. Discriminating high-gain utterances is not the same as routing to reduce WER: the combiner is over-confident on easy utterances that do not need escalation. This is both a null (composite does not beat confidence here) and a methodological lesson (do not trust AUC as a routing proxy).

**EXP-11 (temporal std) — informative but not actionable.** Re-extracting WavLM with per-layer std and adding a temporal-variability feature lifts combiner AUC by **+11 points** (test 0.786) and gives temporal_std the **largest single coefficient** (1.34). But the routing gain is +0.003 area with a CI that includes zero, and — decisively — on the **confidently-hallucinated subset** it provides *no* advantage (the without-temporal combiner is marginally better there). Temporal variability captures **general difficulty**, which confidence already partly captures, not the *escalation-specific* difficulty in confidence's blind spot. The one previously untested branch of the three-way diagnosis (architecture/pooling) thus resolves **negative** and clean.

---

## 6. Results

### 6.1 What "significant" means in this report

Two filters, both required:

- **Uncorrupted** — not an artifact of the pilot's contaminated uncapped-WER metric. The pilot's Pearson r ≈ 0.21, its operating-curve numbers, and its "confidence is worse than random" claim all fail this filter and are excluded from the significant set (they appear only in §6.6, the discard pile, with the reason).
- **Meaningful** — a real, interpretable outcome under the corrected metric, *including clean nulls.* A finding does not have to be a win; "no learned trigger beats free confidence at narrow gaps, and here is why" is a meaningful result. Conversely, a statistically clean number on the wrong quantity (e.g. a strong correlation against a contaminated target) is *not* meaningful.

The significant findings are presented first as a consolidated list (§6.2), then in thematic depth against the two core problems and the cross-cutting headroom strand (§6.3–§6.5). The invalid or non-actionable results are quarantined in §6.6.

### 6.2 Consolidated significant findings

1. **The pilot measured hallucination, not difficulty; capping reverses its conclusions.** (EXP-02 re-score) Under capped WER the pilot probe falls *below random* (AUC 0.32 at τ=0) and confidence rises to the best non-oracle trigger.
2. **Free confidence is a strong baseline — ~47% of the oracle prize, no training.** (EXP-02 re-score) Confidence area 0.0147 vs oracle 0.0311, CI excludes zero.
3. **The routing signal is sparse at the source: median capped gain = 0 for every accent.** (Round 2) Only 47/179 utterances have gain > 0.05; the gain concentrates in Indian English (mean 0.123) with US English slightly *negative* (−0.017).
4. **The probe is not data-limited for this target.** (EXP-05) The learning curve plateaus/declines by 75%; the ceiling is target sparsity, not sample size.
5. **Confidence's advantage is genuine difficulty, not hallucination detection — but it has a defined blind spot.** (EXP-06) Hallucination flags don't subsume it; yet 56% of hallucinations are *confident*, so confidence provably routes them wrong.
6. **A little accent supervision helps; forcing discreteness hurts.** (EXP-08) λ=0.1 is the best learned trigger (0.0108); pushing λ up degrades routing as MI-to-accent rises.
7. **Headroom is manufacturable, and confidence does not scale with it.** (EXP-09) Oracle 0.031→0.058 as the gap widens; confidence's captured *share* falls 47%→16%.
8. **A learned trigger finally beats confidence — but only at the widest gap, and not yet significantly.** (EXP-09) Champion 0.014 vs confidence 0.009 at tiny→large-v3 on 179 utterances.
9. **`turbo` ≈ `large-v3` headroom at lower cost.** (EXP-09) The cheaper careful path is a near-free selective-spend win, independent of trigger quality.
10. **Higher classification AUC does not imply better routing.** (EXP-10, EXP-11) The composite and temporal-std combiners raise AUC substantially while *losing* on area-vs-random — a methodological caution about proxy metrics.
11. **Temporal variability captures general difficulty, not escalation-specific difficulty.** (EXP-11) +11 AUC points, no routing gain, and no help on confidence's blind spot; the architecture branch of the diagnosis resolves negative.

### 6.3 Cross-cutting strand: measurement integrity and headroom (the precondition for both core problems)

Neither core problem is even *askable* until the measurement is trustworthy and there is routing value to measure. This strand is where most of the valid results live, and it reframes the whole project.

**6.3.1 Contamination and the reversal.** Nine of 179 test utterances (5%) have uncapped default WER > 1.0 — Whisper-small generating repetition loops against short references (per-accent uncapped means of 8.96 for Indian and 5.14 for US English are *impossible* as genuine error rates; the careful model's realistic 0.40 on Indian English is the tell). Under Huber δ=0.1 the pilot probe spent its capacity ordering this tail. Capping collapses mean default WER from **2.96 → 0.39** and mean careful WER from **0.76 → 0.34**, and the re-scored leaderboard *inverts*:

| Trigger | Area vs Random (capped) | 95% CI | AUC (τ=0.05) |
|---------|------------------------|--------|--------------|
| Oracle | 0.0311 | [0.020, 0.045] | 0.956 |
| **Confidence** | **0.0147** | [0.003, 0.028] | 0.656 |
| Argmax accent | 0.0103 | [−0.002, 0.023] | 0.515 |
| Scalar probe (pilot) | 0.0035 | [−0.006, 0.013] | 0.427 |
| Random | 0.0000 | [0.000, 0.000] | 0.471 |

The pilot's two loudest claims were both artifacts of the cap boundary.

**6.3.2 The prize is small, and sparse.** The oracle's 0.0311 is the *entire* routing prize on small→large-v3. Median capped gain is **0 for every accent**; only 47/179 utterances have gain > 0.05. On most utterances the two models produce similar capped WER — either both are fine, or both are similarly bad on hard conversational accented speech (large-v3 does not fully rescue conversational Indian or Nigerian English). This is the deep reason the learned signal is weak: *the signal mostly is not there.*

**6.3.3 The power ceiling.** The residual any trigger could win over confidence is ≈ 0.031 − 0.0147 ≈ **0.016**, while paired-bootstrap CIs are ≈ **±0.012** wide at n=179. The setup **cannot statistically distinguish a good learned trigger from free confidence.** This is the pivot from "build a better scalar" (Rounds 1–2) to "manufacture a regime where trigger quality is detectable at all" (Round 3).

**6.3.4 Headroom is manufacturable (the headline grid).** Widening the model gap grows the prize and the signal density:

| Pairing | WER gap | Oracle AVR | Confidence AVR | Champion AVR | n(gain>0.05) |
|---------|---------|-----------|----------------|--------------|--------------|
| small → large-v3 | 0.043 | 0.031 | **0.015** | 0.008 | 47 |
| small → turbo | 0.046 | 0.038 | **0.015** | 0.011 | 40 |
| base → large-v3 | 0.120 | 0.049 | **0.018** | 0.004 | 84 |
| base → turbo | 0.124 | 0.052 | **0.015** | −0.002 | 84 |
| tiny → large-v3 | 0.161 | 0.058 | 0.009 | **0.014** | 98 |
| tiny → turbo | 0.165 | 0.061 | 0.005 | **−0.001** | 94 |

Three things follow. **(a)** Oracle area nearly doubles (0.031 → 0.058). **(b)** Confidence captures a *roughly constant absolute amount* (~0.015) regardless of gap, so its *share* of a growing prize collapses from 47% to 16% — its `avg_logprob` comes from an increasingly weak default model and becomes less informative about when escalation helps. **(c)** `turbo` tracks `large-v3` in headroom at every tier, so the cheaper careful path is a near-free win. The residual (oracle − confidence) grows from 0.016 to 0.049 — real room now exists — but learned triggers capture only a fraction of it.

### 6.4 Core problem 1 — the gradient accentedness signal

**6.4.1 A learned gradient signal exists, but is weak in the narrow regime and does not beat free confidence.** Across EXP-04/05/08 the best learned scalar (multitask λ=0.1) reaches 0.0108 — above argmax, with a CI excluding zero — but its edge over confidence (0.0147) and over the plain gain probe is *not* significant at n=179. The gradient signal is real but marginal where there is little headroom.

**6.4.2 The sharpest gradient-vs-discrete evidence: forcing discreteness hurts.** The multitask λ-sweep is the cleanest on-thesis result. A *small* dose of accent structure (λ=0.1) helps routing; increasing it (λ=0.3, 1.0) *degrades* routing as the scalar's mutual information with accent identity climbs and it collapses toward "argmax in disguise":

| λ | Area vs Random | CI | MI(score, accent) | MI(score, speaker) |
|---|---------------|-----|-------------------|--------------------|
| 0.0 | 0.0047 | [−0.006, 0.017] | 0.244 | 0.306 |
| **0.1** | **0.0108** | [0.003, 0.022] | 0.248 | 0.332 |
| 0.3 | 0.0044 | [−0.006, 0.016] | 0.306 | 0.398 |
| 1.0 | −0.0048 | [−0.015, 0.005] | 0.171 | 0.244 |

This is exactly what the gradient thesis predicts: **degree beats identity.** A continuous difficulty scalar, lightly regularized by accent, routes better than anything pulled toward hard categories.

**6.4.3 The regime where the gradient signal earns its keep is *located.*** EXP-09 shows the learned trigger crossing confidence only once headroom is manufactured (tiny→large-v3: 0.014 vs 0.009). This is the project's thesis, resolved conditionally: *the gradient scalar earns its keep in the wide-gap regime, not the narrow one.* The crossing is not yet statistically robust on 179 utterances — establishing significance is a test-power problem (§7.6), not a modelling one.

**6.4.4 Confidence validates the "scalar as quality signal" half of the thesis — for free.** The strongest label-free gradient trigger in the narrow regime is not the learned probe; it is Whisper's own confidence, itself a continuous difficulty scalar. That both vindicates the gradient framing (a continuous quality signal *is* the right routing object) and undercuts the *learned* version's marginal value (the free scalar already captures ~47%). The learned scalar's distinctive value must therefore come from the *within-accent, non-hallucination* gradient it adds beyond confidence — which only becomes visible when the gap is wide.

**6.4.5 The scalar is not a speaker recognizer collapse — but leans speaker over accent.** Pilot leakage (MI speaker 0.32 vs accent 0.22, ratio 1.44) and the multitask MI columns show the scalar correlates *more* with speaker than accent. Under speaker-held-out evaluation this is defensible (utterance difficulty genuinely varies by speaker), but it flags that the scalar is partly a speaker-difficulty estimator, which bounds generalization to unseen speakers.

### 6.5 Core problem 2 — the unbounded taxonomy problem

**6.5.1 The discrete argmax baseline is strong *only because the set is closed.*** Argmax-accent reaches 0.0103 — within CI of confidence — which looks like a win for the taxonomy-bound approach. But EXP-06's slice analysis shows *why*: escalation gain is concentrated in Indian English (mean 0.123), with several accents at ~zero gain and US English slightly negative. On a fixed 6-accent set, "always escalate the one or two high-gain groups" captures most of the available prize. **This is precisely the closed-set advantage that the unbounded-taxonomy argument predicts will not survive distribution shift:** the moment the deployment accent mix differs, or an unseen accent appears, a group-level rule has no group to route.

**6.5.2 Label-free triggers match the discrete baseline without enumerating accents.** Confidence (0.0147) and the learned scalar (0.0108) equal or exceed argmax (0.0103) while never naming an accent. On this data the label-free approach loses nothing for its generality — and it carries the open-set robustness the discrete baseline lacks. That is the taxonomy thesis's core claim, borne out (though, honestly, not *stress-tested* here — see 6.5.4).

**6.5.3 Forcing the scalar to be taxonomy-aware degrades it.** The same λ-sweep (6.4.2) is *also* a taxonomy result: as the scalar is pushed to encode discrete accent identity, routing gets worse. Enumerating the taxonomy inside the trigger is actively harmful, not merely unnecessary.

**6.5.4 What is *not* yet shown.** The taxonomy thesis is vindicated *in principle* on a closed set, but the decisive experiment — evaluating on **held-out unseen accents** (e.g. AfriSpeech-200's 41 test-only accents) where a discrete classifier falls off its label set and a scalar should degrade gracefully — was out of scope. This is the single most important open test of core problem 2 (§7.6).

### 6.6 The discard pile — results that are *not* valid findings, and why

Honesty about what to *ignore* is part of the deliverable.

- **Pilot Pearson r ≈ 0.21** — measured against uncapped, hallucination-contaminated default WER. It reflects hallucination-ranking, not accent difficulty. **Not a valid signal.**
- **Pilot "confidence worse than random" (area −1.33)** — an artifact of uncapped WER letting a few blow-ups dominate the integrated area. Under capping, confidence is the *best* non-oracle trigger. **Reversed.**
- **Pilot operating-curve numbers (oracle 0.65, scalar 0.45, etc.)** — on the uncapped scale; not comparable to any capped number and not trustworthy. **Superseded.**
- **Capped-WER probe val Pearson 0.675 (EXP-04)** — statistically clean and genuinely better training stability, but it did **not** translate into routing advantage. Meaningful for *diagnosis* (the target can be regressed), null for the *product* (routing). Reported as such, not as a win.
- **Composite test AUC 0.78 (EXP-10) and temporal-std +11 AUC (EXP-11)** — real classification gains that **do not** improve routing area-vs-random. High AUC here is a *distractor*, not a result; the valid finding is the AUC–area disconnect itself.
- **EXP-07 stats-pooling (R2)** — infrastructure only; produced no scored result until realized as EXP-11.
- **EXP-12 accent-adapted path** — deferred behind a verify-before-trust guard; produced no result. It is a *plan*, not a finding.

---

## 7. Discussion

### 7.1 The three-round arc, read as one argument

| Round | Question | Answer |
|-------|----------|--------|
| Pilot | Can a learned scalar route better than heuristics? | Inconclusive — the evaluation was contaminated by hallucination. |
| Round 2 | What is the binding constraint? | **Headroom.** The oracle ceiling is too low and the target too sparse for triggers to differentiate; the setup is underpowered. |
| Round 3 | Does more headroom help? | Oracle grows, confidence's *share* shrinks, a learned trigger finally crosses confidence at the widest gap — but inconsistently, and not yet significantly. |

The arc is a single scientific story: **contamination fixed → headroom identified as binding → regime located where trigger quality becomes detectable.** Each round's *method* changed in response to the previous round's *finding*, which is the point — the value of the work is the diagnosis, not a leaderboard win.

### 7.2 What this means for the two core problems

- **Gradient signal (problem 1): conditionally vindicated.** The right routing object *is* a continuous difficulty scalar — confidence proves the free version works, and the multitask sweep proves that *degree* beats *identity*. The *learned* scalar's marginal value over free confidence is real but only surfaces once the model gap is wide. The honest statement: "a learned accentedness scalar is worth building when you have manufacturable headroom (a genuinely weak default vs a strong careful model); below that, use free confidence."
- **Unbounded taxonomy (problem 2): vindicated in principle, untested at the limit.** Label-free triggers match the discrete baseline without enumerating accents, and forcing the trigger toward a taxonomy *hurts.* But the closed 6-accent set flatters the discrete baseline; the decisive open-set test is future work.

### 7.3 The production takeaway (the "when do you actually need accent-aware routing?" answer)

This is exactly Deepgram's own framing (accent-robust default first; add accent-specific machinery only past precision and ROI gates), and the experiments land on the same stance from the data side:

1. **Narrow model gap → use free confidence.** It captures ~half the oracle prize, needs no training and no feature extraction, and no learned trigger significantly beats it.
2. **Wide model gap → a learned or composite trigger can earn its keep**, but only where headroom demonstrably exists, and the advantage should be *confirmed on a larger test set* before it is trusted.
3. **Prefer `turbo` as the careful path** — near-identical headroom to `large-v3` at lower cost, independent of trigger quality.
4. **Do not trust AUC as a routing proxy** — optimize and select on area-vs-random (actual WER reduction), because high AUC combiners lost on routing here.

### 7.4 Threats to validity and limitations

- **Statistical power.** n=179 test utterances, with the routing-relevant subset often < 100. Several key comparisons (champion vs confidence at wide gap; λ=0.1 vs gain probe; temporal std) return CIs that include zero. Nothing here should be read as a *significant win* of a learned trigger over confidence — only as a located regime and a direction.
- **Single corpus, closed accent set.** All results are on a 6-accent EdAcc subset. Generalization across corpora, channels, and unseen accents is asserted by design, not demonstrated.
- **Speaker leakage.** The scalar leans on speaker identity more than accent; speaker-held-out splits make this defensible but not eliminated.
- **Careful model imperfect.** large-v3 does not fully rescue conversational Indian/Nigerian English, which caps the gain even where both models are weak — a property of the *models*, not the router.
- **Combiner comparability caveat.** The two combiner evaluations (EXP-10 and EXP-11) use different feature sets and are not directly comparable to each other; each is valid within its own experiment.

### 7.5 What I would do with another week — ranked

1. **Run the accent-adapted careful path (EXP-12), gated by verify-before-trust.** This is the strongest on-thesis next step and the only one that directly tests "try *differently*, not just harder." Route the Indian-English slice (mean gain 0.123) to `Tejveer12/Indian-Accent-English-Whisper-Finetuned`; *first* confirm it beats general large-v3 on that slice, and stop if it does not. If it opens Indian-slice headroom a bigger *general* model cannot, that is concrete evidence for accent-*aware* routing over pure quality escalation — the headline "when you actually need accent-awareness" finding.
2. **Open-set / unseen-accent evaluation (the decisive taxonomy test).** Evaluate the label-free scalar against the discrete argmax classifier on held-out accents (AfriSpeech-200's 41 test-only accents, or EdAcc accents withheld from the argmax label set). The gradient/label-free thesis predicts the classifier falls off its taxonomy while the scalar degrades gracefully. This is the experiment that would *stress-test* core problem 2 rather than merely illustrate it.
3. **Scale test power at a wide gap.** The champion's wide-gap edge (0.014 vs 0.009) is a power problem. A 5–10× larger, accent-diverse test set at tiny→large-v3 — plausibly via a scoped Common Voice pull *combined with* the wide gap (Round 2 showed data alone doesn't help at the narrow gap) — would resolve whether the crossing is real.
4. **A genuinely decision-aligned target with denoising.** Regress escalation gain computed with careful-model transcripts, but first *denoise* the target: separate the hallucination-vs-not decision (a binary auxiliary head) from the graded within-accent difficulty, so the probe is not asked to order a spike-plus-tail distribution.
5. **Richer pooling that is escalation-specific, not difficulty-general.** The temporal-std null suggests mean+std captures *general* difficulty. Attention-over-time or segment-level features targeted at the *confidently-hallucinated* subset (confidence's proven blind spot) are the principled next architectural move — but only worth it once headroom exists.
6. **Operationalize the drift-detector.** The scalar's distribution is a monitoring signal by construction. Simulate an accent-mix shift or inject a read-speech batch and show the distribution flags it — closing the flywheel loop (eval outputs → next iteration) that the role explicitly values.

---

## Appendix A — Workflow: How This Was Built

The project is as much a demonstration of *research process under time budget* as of a result. The process had four distinct stages, and the AI-collaboration posture is described last.

### A.1 Stage 1 — Scoping the problem (~1 hour of reading and structuring)

Before any code, the ambiguous one-line assignment was turned into a precise, defensible problem by building a small knowledge base and reasoning from it. This had three moves:

- **Learn the state of the art.** Survey what the field has actually built for accent handling, and organize it so later decisions point back to a specific place in the landscape — the accent-agnostic/aware × single-model/routing 2×2, the near-empty "quality-driven escalation as an accent strategy" cell, and the literature's habit of reporting specialist WER rather than *net* WER (`accent-routing-sota.md`).
- **Understand what accent actually is** — but only to the depth that changes an engineering decision: the decomposition of accent into weak cues, the "survives the text bottleneck?" column that forces the trigger to be acoustic/SSL, the layer-distribution that motivates a learnable weighted sum, and the speaker/channel confound that dictates held-out evaluation (`what-is-accent.md`).
- **Convert insight into scope.** Fold both into a single scoping document that states, unambiguously, what the problem is (a routing decision, not a label), what a solution is (a quality-vs-cost operating curve), and what constrains it (pretrained-first, ~5 hours, 24 GB Apple Silicon, EdAcc, net WER including misrouting) — deliberately defining the *design space* rather than committing to one instantiation (`accent-routing-scoping.md`).

The payoff of this stage is that every later decision — capped WER, net-WER-not-accuracy, SSL-not-transcript, scalar-not-argmax — traces back to a written rationale, which is what makes the artifacts defensible under interview questioning.

### A.2 Stage 2 — Candidate generation and selection

With the design space defined, generative AI was used to instantiate it into **six concrete problem/solution proposals**, each specifying a target, trigger, baseline, improvement, and data choice, all required to satisfy the scoped definition and draw only from the in-scope mechanisms. These were interrogated (feasibility in the time/hardware budget; what each would prove; how each is baselined) rather than accepted. The **accentedness scalar** was chosen because it uniquely (a) fits the budget with only a small probe training, (b) occupies the sparse, interesting right-hand cell of the landscape, (c) addresses *both* core problems — gradient signal and unbounded taxonomy — at once, and (d) doubles as a drift/monitoring signal. The discrete argmax classifier was kept as the baseline to beat, not the contribution.

### A.3 Stage 3 — Spec-driven development via planning prompts

Rather than prompting ad hoc, each round of experimentation was driven by a **single, detailed planning prompt** handed to Claude Code that (i) stated the objective and the non-negotiable comparability rules, (ii) specified a phased plan with acceptance checks, (iii) required a written implementation plan approved *before* any code (Gate 0), and (iv) pre-registered a **decision rule** for every experiment mapping each possible outcome — including nulls — to a conclusion and a next action. Three such prompts were written: the pilot spec, the Round-2 five-extension ablation, and the Round-3 headroom sweep.

Two properties of this style are load-bearing:

- **Pre-registered decision rules** mean a null result is interpretable rather than a dead end — the prompt already says what "flat learning curve" or "composite ≈ confidence" *means*. This is why Round 2's mostly-null ablation is a strong result rather than a failure.
- **Writeup-ready at every gate** — each experiment writes a self-contained report (date/commit, question, config, method, results-with-CIs, the filled-in decision rule, caveats, next action), so stopping after any phase still yields a coherent deliverable.

### A.4 Stage 4 — The pilot → Round 2 → Round 3 loop

Execution ran as an autonomous Claude Code session per round, pausing only at hard gates (large downloads, compute-heavy re-extraction, unverified model downloads, or material design deviations). Each round was *caused* by the previous round's finding:

- **Pilot** surfaced the problems: it stood up the pipeline and, critically, exposed the hallucination contamination and the weakness of the raw signal.
- **Round 2** diagnosed them: a shared corrected harness (capped WER, AUC/AP, bootstrap CIs) re-scored the pilot — reversing its conclusions — then a five-way ablation localized the bottleneck to headroom/target-sparsity rather than data or (as far as tested) architecture.
- **Round 3** acted on the diagnosis: instead of chasing a better trigger inside a fixed problem, it manipulated the problem (widened the model gap) to find the regime where trigger quality is detectable, and tested composite and temporal features against the free-confidence bar.

Discipline throughout: numbered `EXP-NN` folders with dated reports, a living `COMPARISON.md` leaderboard, a frozen test fold, a single shared scorer, `DECISIONS.md` and `OPEN-QUESTIONS.md` for asynchronous review, and an `ai-use-log.md`. (A numbering collision — a pre-existing `EXP-03` — was caught and corrected mid-session, which is the kind of small integrity check the folder discipline is meant to force.)

### A.5 AI-collaboration posture (delegated / accepted / corrected / rejected / validated)

- **Delegated:** repo scaffolding, environment setup, data prep and speaker-disjoint splitting, ASR/feature caching, baseline and probe implementation, plotting, experiment-report drafting, and debugging.
- **Accepted** only against acceptance checks and pre-registered decision rules — never because the model asserted success.
- **Corrected / rejected:** the contaminated uncapped-WER metric was the biggest correction (it inverted the pilot's story); AUC was rejected as a routing proxy once it diverged from area-vs-random; and the accent-adapted model was *not* trusted without a verify-before-trust WER check on its own slice.
- **Validated** via the frozen test fold, one shared scorer, bootstrap CIs, unit tests on the metric code, and — most importantly — by treating every headline claim as something that must survive a paired bootstrap against the *free* baseline (confidence), not just look good in isolation.

The through-line: generative AI did the building and much of the drafting; the human owned the framing, the metric integrity, the decision rules, and the judgment about which results to *believe.*

---

## Appendix B — Consolidated Result Tables

### B.1 Round-2 final leaderboard (frozen test fold, capped WER, ranked by area vs random)

| Rank | Trigger | Area vs Random | Significant (CI excl. 0)? | Source |
|------|---------|---------------|---------------------------|--------|
| 1 | Oracle | 0.0311 | Yes | Upper bound |
| 2 | **Confidence** | **0.0147** | **Yes** | Whisper `avg_logprob` |
| 3 | Multitask (λ=0.1) | 0.0108 | Yes | Learned |
| 4 | Argmax accent | 0.0103 | No | Group-level (discrete taxonomy) |
| 5 | Hallucination union | 0.0075 | No | Heuristic |
| 6 | No-speech prob | 0.0069 | No | Whisper metadata |
| 7 | Probe (gain target) | 0.0058 | No | Learned |
| 8 | Probe (capped WER) | 0.0055 | No | Learned |
| 9 | Multitask (λ=0.0) | 0.0047 | No | Learned |
| 10 | Scalar probe (pilot) | 0.0035 | No | Learned |
| — | Random | 0.0000 | — | Baseline |
| — | Compression ratio | −0.0053 | No (worse than random) | Heuristic |

### B.2 Round-3 headroom grid (six pairings)

*(reproduced from §6.3.4 for reference: WER gap, oracle/confidence/champion area-vs-random, signal density)*

| Pairing | WER gap | Oracle | Confidence | Champion | n(gain>0.05) | Confidence share |
|---------|---------|--------|-----------|----------|--------------|------------------|
| small → large-v3 | 0.043 | 0.031 | 0.015 | 0.008 | 47 | 47% |
| small → turbo | 0.046 | 0.038 | 0.015 | 0.011 | 40 | 39% |
| base → large-v3 | 0.120 | 0.049 | 0.018 | 0.004 | 84 | 37% |
| base → turbo | 0.124 | 0.052 | 0.015 | −0.002 | 84 | 29% |
| tiny → large-v3 | 0.161 | 0.058 | 0.009 | **0.014** | 98 | 16% |
| tiny → turbo | 0.165 | 0.061 | 0.005 | −0.001 | 94 | 8% |

### B.3 Per-accent gain structure (capped, small → large-v3)

| Accent | n (test) | Hallucinated | Capped mean WER (default) | Mean capped gain | Routing helps? |
|--------|----------|--------------|---------------------------|------------------|----------------|
| Indian English | 35 | 3 | 0.50 | **0.123** | Yes (the prize) |
| US English | 27 | 3 | 0.45 | −0.017 | No (careful sometimes worse) |
| Southern British | 28 | 1 | 0.38 | small + | Marginal |
| Scottish | 37 | 1 | 0.30 | ~0 | No |
| Nigerian | 30 | 1 | 0.36 | ~0 | No |
| Irish | 22 | 0 | 0.31 | ~0 | No |

*Median capped gain is 0 for every accent; the positive prize is concentrated in Indian English.*

### B.4 Multitask λ-sweep (EXP-08)

| λ | Area vs Random | CI | MI(score, accent) | MI(score, speaker) | Reading |
|---|---------------|-----|-------------------|--------------------|---------|
| 0.0 | 0.0047 | [−0.006, 0.017] | 0.244 | 0.306 | plain gain probe |
| **0.1** | **0.0108** | [0.003, 0.022] | 0.248 | 0.332 | best learned trigger |
| 0.3 | 0.0044 | [−0.006, 0.016] | 0.306 | 0.398 | starts encoding accent → degrades |
| 1.0 | −0.0048 | [−0.015, 0.005] | 0.171 | 0.244 | pure classifier → worse than random |

---

## Appendix C — Glossary (quick reference)

- **WER** — word error rate, `(S+I+D)/N_ref`; unbounded above.
- **Capped WER** — `min(WER, 1.0)`; applied before all aggregation and target construction from Round 2 on; neutralizes hallucination blow-ups.
- **Default-model WER (pilot target)** — the raw WER of the fast model; the pilot regressed this and was contaminated by hallucination.
- **Escalation gain (Round-2+ target)** — `cap_wer(default) − cap_wer(careful) ∈ [−1,1]`; what escalation *buys*; sign kept (negative = careful worse). The decision-aligned target.
- **Operating curve** — net WER vs escalation rate as the trigger threshold sweeps; misrouting cost included automatically.
- **Escalation rate** — fraction of utterances sent to the careful path; the compute/$ cost proxy.
- **Area over random (`area_vs_random`)** — `∫ (netWER_random − netWER_trigger) dr` over escalation rate; random = 0 by construction, oracle sets the ceiling; positive = better than random. Scale depends on capping (uncapped and capped values are not comparable).
- **Net WER @ budget** — the operating curve's height at a fixed escalation rate.
- **AUC / AP (`gain > τ`)** — ranking-quality scorecard for "escalation helps by more than τ" (τ = 0, 0.05); measures ordering independent of threshold. **Not a reliable routing proxy** (see the AUC–area disconnect).
- **Oracle / Random** — upper / lower reference routers.
- **Confidence trigger** — route on the default model's `avg_logprob`; default-specific; the free baseline to beat.
- **Argmax-accent** — route whole accent groups by group-level WER/gain; the discrete-taxonomy baseline.
- **WER gap** — `mean(cap_wer_default) − mean(cap_wer_careful)` per model pairing; the headroom knob swept in Round 3.
- **Signal density** — count of utterances with capped gain > 0.05; how many utterances routing can act on.
- **Headroom** — the total routing prize available, i.e. the oracle's area over random.
- **MI(score, ·)** — mutual information of the scalar with speaker vs accent identity; the leakage / argmax-in-disguise check.
- **Paired bootstrap** — CI on the *difference* between two triggers on resampled utterances; the correct test for "A beats B."
