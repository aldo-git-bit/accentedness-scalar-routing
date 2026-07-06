# Claude Code Planning Prompt — Round 2: From Pilot to Better Models (Accentedness-Scalar Routing)

> **How to use this prompt.** Paste this into Claude Code in the existing `accentedness-scalar-routing` repo. **First produce a written implementation plan** (phase-by-phase task breakdown + the new files/targets you will add) and stop for my review (**Gate 0**). Do not start coding until I approve.
>
> This is a **single, phased second round**, not five separate jobs — the extensions share one evaluation harness and one frozen test fold, and their *conclusions* are sequentially dependent (Ext 2 tells us whether Ext 1 is data-starved; Ext 1 fixes the metric everything else is scored against; Ext 3 may reframe whether the scalar's edge survives). Running them as isolated prompts would re-derive shared context five times and risk implementing the metric five different ways, which would destroy cross-experiment comparability — the entire point of this round.
>
> **I have chosen to gate after every extension.** After the plan is approved, work autonomously *within* a phase, but **hard-stop and wait for me at the end of Phase 0 and at the end of each extension** (see §0.1). At each stop, present the short gate report in §8 so I can approve or redirect quickly.

---

## 0. Context — what changed since Round 1

Round 1 built the pilot: a ~263K-param probe on WavLM-large mean-pooled features, regressing Whisper-small per-utterance WER (Huber δ=0.1), routing small→large-v3 on EdAcc (6 accents × 150 = 900 utts). It works but the result is soft: the probe beats argmax-accent at 20–30% escalation but doesn't dominate, layer weights are near-uniform, and val Pearson r ≈ 0.21 (below the 0.3 target). See `writeup.md`.

**This round transitions from "does a scalar work at all" (pilot) to "what is the scalar actually worth, and why is the signal weak" (better models).** Before any new modeling, Phase 0 writes a diagnosis doc that records the reasoning behind this transition, then builds a shared, corrected evaluation harness that all five extensions import. The extensions then resolve, in priority order, whether the weak signal is a **target** problem, a **data** problem, or an **architecture** problem — and whether the scalar's advantage over trivial baselines is real.

The knowledge-base docs (`docs/what-is-accent.md`, `accent-routing-sota.md`, `accent-routing-scoping.md`, `six-proposals.md`) remain the source of truth for framing. This round does **not** change scope: still cascaded English ASR, trigger from SSL/acoustics never transcript, pretrained-first (only the small probe trains), report curves not points.

### 0.1 Gate structure (STOP and wait for me at each of these)

| Gate | When | What I approve |
|---|---|---|
| **Gate 0** | After you write the plan, before any code | The plan itself |
| **Gate A** | End of Phase 0 | Diagnosis doc + shared harness + **re-scored pilot/baselines under corrected metrics** (this may change the pilot's conclusions — I want to see that before spending budget) |
| **Gate 1** | End of Extension 1 (EXP-04) | Gain-target result + decision-rule verdict |
| **Gate 2** | End of Extension 2 (EXP-05) | Learning-curve + accent-probe diagnostic verdict (this decides whether Common Voice is worth it) |
| **Gate 3** | End of Extension 3 (EXP-06) | Hallucination baselines + confidence autopsy verdict |
| **Gate 4** | End of Extension 4 (EXP-07) | Stats-pooling result. **Also pause *before starting* Ext 4** to confirm the WavLM re-extraction (the only compute-heavy step this round) |
| **Gate 5** | End of Extension 5 (EXP-08) | Multi-task result |
| **Final** | After synthesis | `writeup-v2` + updated leaderboard |

I expect to reach Extensions 1–3 for sure; 4 and 5 are stretch. Order is priority order — **do not reorder**, and do not skip ahead to a later extension to "save time."

### 0.2 Working autonomously *within* a phase

Between the gates, default to action: choose sensible defaults, log them in `docs/DECISIONS.md`, park non-blocking questions in `docs/OPEN-QUESTIONS.md`, and keep going. Debug and fix your own errors; don't stall on anything that isn't a gate. Keep `docs/ai-use-log.md` current. The only *unplanned* interrupts (beyond the gates above) are the standard ones: a download >~15 GB or an asset not named here; training heavier than the small probe; or an unrecoverable credential/access block. None of Extensions 1–3 or 5 should need any download or heavy training; Extension 4 needs only a local CPU re-extraction (no download).

---

## 1. Objective of Round 2

Clarify **the marginal value of the accentedness scalar** and **the cause of its weak learned signal**, by:

1. aligning the probe's target and the evaluation metric with the actual routing *decision* (Ext 1);
2. localizing the bottleneck — target noise vs. data volume vs. architecture/pooling (Ext 2, and Ext 4);
3. stress-testing whether trivial signals (hallucination flags, confidence) already capture the scalar's gains (Ext 3);
4. testing whether accent supervision helps or merely collapses the scalar back toward the argmax baseline (Ext 5).

Every extension must end with a **decision rule already written down** (in this prompt) mapping its outcome to "what this means and what to do next," so the result is interpretable even if it's negative.

---

## 2. Non-negotiable comparability rules

These are what make five experiments a *progression* rather than five islands. Violating any of them silently corrupts the whole round.

1. **Frozen test fold.** Every extension evaluates on the **identical speaker-disjoint test fold** from the pilot split manifest (same seed, same speakers). **Do not re-split.** Learning-curve subsampling (Ext 2) subsamples the **train fold only**, and subsamples **by speaker** (never by utterance) to preserve speaker-disjointness.
2. **One shared scorer.** All triggers — old and new — are scored through the single `src/accentedness_routing/eval/eval_common.py` built in Phase 0. No extension may fork the curve/metric logic. If a metric needs to change, it changes in `eval_common.py` and *every* result is re-scored.
3. **Capped WER everywhere.** All WER is clipped to 1.0 before aggregation and before use as a probe target. Report **both micro and macro** (per-accent-averaged) aggregates; the pilot's headline numbers were micro and outlier-dominated.
4. **Curves with confidence, not points.** Primary output is the net-WER-vs-escalation-rate operating curve with a **bootstrap 95% CI band** (resample utterances, ≥1000 draws). Summary scalars (netWER@10/20/30/50%, area-vs-random) and the probe scorecard all carry bootstrap CIs. Pairwise claims ("scalar beats argmax at 20%") must report a **paired bootstrap CI on the difference**, not two overlapping bands.
5. **Carry the reference set forward.** Every comparison figure/table always includes the same reference triggers: **oracle, random, argmax-accent, confidence, and the frozen pilot scalar.** New triggers are added on top so each extension is read against a fixed backdrop.
6. **Determinism.** Fixed seeds; log key lib versions into every report.

---

## 3. Phase 0 — Diagnosis doc + shared harness + re-scored canonical baselines (→ **Gate A**)

Three deliverables. This phase does **no new modeling** — it corrects the lens and records the reasoning, then re-scores what already exists.

### 3.1 `docs/DIAGNOSIS-pilot-to-v2.md` (the record of the transition)

A critical assessment of Round 1 that states plainly why we're moving to better models. It must make the following arguments (these are the substance — write them out, don't just list them), and explicitly expand each of the seven pilot Failure Modes/Limitations under this lens:

- **WER contamination / hallucination artifact.** Per-accent *mean* default WERs >1.0 (Indian 8.96, US 5.14) are not "difficulty" — WER exceeds 1.0 only via massive insertion, i.e. Whisper-small hallucinating repetition loops against short references. The careful model's ~0.40 on Indian English is plausible; the 8.96 is an artifact. Consequence: the pilot probe regresses these raw numbers with Huber δ=0.1, so the loss is dominated by ordering a hallucination tail, and the aggregate operating-curve WER is set by a few blow-ups rather than by routing quality. **This motivates capped WER (rule §2.3) and is the first thing Ext 1 fixes.**
- **Effective signal density ≪ 544.** Three of six accents (Irish, Scottish, Nigerian) have near-zero escalation gain, so ~half the training data teaches "don't escalate, and it barely matters"; within the signal-carrying half, Indian-English hallucination cases dominate. The *effective* learnable signal is far smaller than the raw utterance count — which is why "just add more EdAcc-like data" is not obviously the fix.
- **Why "just add Common Voice" is not the first move.** More EdAcc-like data mostly adds more near-zero-gain utterances; CV additionally introduces a read-vs-conversational domain shift on top of the accent question. Whether data volume is even the bottleneck is an *empirical question Ext 2 answers first* (learning curve), before we pay the CV tax.
- **Metric critique.** Pearson r on a spike-at-zero-plus-heavy-tail target is nearly uninterpretable, and routing is a ranking/threshold problem, so the honest scorecard is **AUC/AP for "escalation helps by >τ"** plus bootstrap CIs on the curve. The pilot's 1.00-vs-1.28 edge over argmax, on 179 test utts concentrated in 2–3 accents, is very possibly not significant — establishing that is itself a result.
- **The three-way uncertainty this round resolves.** Is the weak signal a **target** problem (regressing the wrong, noisy quantity), a **data** problem (too few signal-carrying utterances to specialize layers), or an **architecture** problem (mean-pooling / weighted-sum too weak)? Map each extension to which hypothesis it tests (Ext 1 → target; Ext 2 → data vs. target vs. architecture, via learning curve + accent-probe control; Ext 3 → is the gain even non-trivial; Ext 4 → pooling/architecture; Ext 5 → does accent structure help or just reproduce argmax).

### 3.2 `src/accentedness_routing/eval/eval_common.py` (the one shared harness)

Consolidate/extend the existing eval into a single canonical module every extension imports. It must provide:

- `cap_wer(w)` → `min(w, 1.0)`; applied before all aggregation and all target construction.
- `escalation_gain(wer_default, wer_careful)` → `cap_wer(wer_default) - cap_wer(wer_careful)` ∈ [−1, 1] (careful can be worse → negative gain; keep the sign).
- `operating_curve(scores, wer_default, wer_careful)` → net WER (careful where escalated, default elsewhere, misrouting cost included) vs escalation rate over a threshold sweep, micro **and** macro.
- `summarize(curve)` → netWER@{10,20,30,50}%, area-between-curve-and-random (normalized).
- `decision_scorecard(scores, gain, tau)` → **AUC and AP** for the binary event `gain > tau` (default τ=0.0 and a small positive τ, e.g. 0.05), plus Spearman (report Pearson too for continuity with the pilot).
- `bootstrap(...)` → 95% CIs (≥1000 utterance resamples) on every summary scalar and a CI band on the curve; and `paired_bootstrap(trigger_a, trigger_b)` → CI on the difference at fixed budgets.
- Unit tests in `tests/` (e.g. capping behavior; oracle ≥ every trigger; random ≈ diagonal within CI).

### 3.3 Re-score everything that already exists, and build the comparison backbone

- Re-score **all pilot triggers** (oracle, random, argmax, confidence, pilot scalar) through the new harness with capped WER + CIs. This produces the **canonical reference set** used by every later extension.
- Create `experiments/COMPARISON.md` — a living leaderboard, one row per trigger/experiment, all on the frozen test fold under the corrected metrics: netWER@{10,20,30,50}% (with CI), area-vs-random (with CI), AUC/AP, and a one-line "what it is." Seed it with the re-scored reference set.
- Add `make compare` → regenerates a master overlay figure (`experiments/figures/operating_curves_all.png`) of every trigger's curve with CI bands, and rebuilds `COMPARISON.md`.
- Update `experiments/INDEX.md`.

**Gate A.** Present: the diagnosis doc; the re-scored reference table (did capping/CIs change the pilot's story — especially whether the scalar still beats argmax significantly?); and the seeded `COMPARISON.md`. **Stop.**

---

## 4. The Extensions (each is one experiment; priority order; gate after each)

Naming: continue the existing `EXP-<NN>` sequence *and* encode the extension number in the slug, so the counter stays coherent and the extension label is explicit. The `experiments/` folder already contains `EXP-01`, `EXP-02`, and **`EXP-03-flywheel`**, so this round starts at **EXP-04**.

| Ext | Folder | One-liner | New cost |
|---|---|---|---|
| 1 | `experiments/EXP-04-extension1-gain-target/` | Regress clipped escalation-gain, not default-WER | none (cache) |
| 2 | `experiments/EXP-05-extension2-diagnostics/` | Learning curve + accent-probe control | none (cache) |
| 3 | `experiments/EXP-06-extension3-hallucination/` | Hallucination-flag baselines + confidence autopsy | none (cache) |
| 4 | `experiments/EXP-07-extension4-stats-pooling/` | mean+std over time pooling | **WavLM re-extraction (~2×)** |
| 5 | `experiments/EXP-08-extension5-multitask/` | WER/gain + accent multi-task probe | none (cache) |

Each extension's `report.md` follows §7 and **must** contain the pre-registered **Decision rule** block quoted below for that extension, filled in with the observed outcome, plus a one-row append to `COMPARISON.md` and an `INDEX.md` update.

---

### Extension 1 — Gain-target probe + decision-aligned scoring (EXP-04) → Gate 1

**Question.** Does regressing **clipped escalation-gain** instead of default-WER produce a better router, and does it change the pilot's conclusions?

**Method (all from cache; no ASR re-run).**
- Target = `escalation_gain` per utterance from the two cached model outputs.
- Retrain the **identical** ~263K architecture and training recipe (same cached mean-pooled features, same speaker-disjoint splits, same AdamW/lr/early-stopping), changing **only the target**. Huber loss on the bounded gain target.
- Controlled comparison: also retrain the **default-WER target under capped WER** in the identical harness (the pilot used uncapped Huber), so "target" is the only variable vs. the pilot replication.
- Score both through `eval_common`; report the operating curve, scorecard (AUC/AP for gain>τ), and **paired bootstrap CIs vs. argmax and vs. the pilot scalar**.

**Deliverables.** `models/probe_gain.pt`; the two-probe comparison (capped-WER-target vs gain-target) against the reference set; report; COMPARISON rows.

**Acceptance.** Gain-target curve computed on the frozen test fold on the shared axes with CIs; scorecard reported.

**Decision rule (pre-registered).**
- Gain-target **dominates or matches argmax with a tighter/│significant│ CI** → the scalar's value is clarified upward; the wrong-target hypothesis is confirmed. Adopt gain-target as the champion probe for Ext 2/4/5.
- Gain-target **no better than the capped-WER probe, both ≈ argmax** → the ceiling is not the target. Proceed to Ext 2 to test data vs. architecture; do not attribute the weak signal to the target any further.

---

### Extension 2 — Learning-curve + accent-probe diagnostic (EXP-05) → Gate 2

**Question.** Is the weak signal a **data-volume** limit or a **target-noise / architecture** limit? (This decides whether Common Voice is worth pursuing.)

**Method (all from cache; two sub-experiments).**
- **2a Learning curve.** Train the champion probe (best target from Ext 1) on **25/50/75/100%** of the train fold, subsampled **by speaker**. Plot val AUC (and r) vs. train size with CIs. Slope at 100% is the read-out.
- **2b Accent-probe control.** Train the **same architecture** to classify standardized accent from the **same** features. Report accuracy, macro-F1, and — crucially — the **learned layer-weight distribution**. This is the clean control: if accent-ID specializes its layer weights and classifies well while the WER/gain probe's weights stay uniform, then near-uniform weights are a property of the **noisy regression target**, not a capacity/data ceiling.

**Deliverables.** Learning-curve figure; accent-probe accuracy + layer-weight comparison figure (accent-probe vs gain-probe weights side by side); report; COMPARISON note (accent-probe is diagnostic, not a router — mark it clearly).

**Acceptance.** Learning curve with CIs; accent-probe metrics + both layer-weight vectors reported.

**Decision rule (pre-registered).**
- **LC still climbing at 100% + accent-probe specializes/classifies well** → data-limited on a *learnable* target → Common Voice is justified. Flag a scoped CV pull as the top "next" item (a CV download is a gate — do not pull it autonomously; recommend it at Gate 2).
- **LC flat by ~75% + accent-probe specializes** → the **target** is the ceiling (per-utterance gain is too noisy to learn from these features); more data won't help. Honest headline; consider target denoising / a cleaner label as future work.
- **LC flat + accent-probe *also* fails to specialize** → **architecture/feature** ceiling → Ext 4 (pooling) is the indicated move; data is not the story.

---

### Extension 3 — Hallucination-aware baselines + confidence autopsy (EXP-06) → Gate 3

**Question.** Do trivial signals already capture the scalar's gains — i.e. is the routing benefit just "catch the hallucinations"? And why is the confidence baseline worse than random (area −1.33)?

**Method (all from cache; no ASR re-run).**
- **compression_ratio** computed post-hoc from cached hypothesis text: `len(text.encode()) / len(zlib.compress(text.encode()))`. This is faithful to Whisper's own hallucination signal (text gzip ratio); note in DECISIONS the only difference is per-utterance (final hypothesis) vs Whisper's per-segment application.
- **no_speech_prob** used directly from cache.
- Build hallucination-flag routing triggers: (i) compression_ratio > θ, (ii) no_speech_prob > θ, (iii) union. Score→curve through `eval_common`. Add as **new reference baselines** carried forward.
- **Confidence autopsy.** Verify the sign convention (low avg_logprob → escalate). Plot avg_logprob vs capped WER overall and on the hallucination subset (WER>0.5). Test the hypothesis that Whisper hallucinates *confidently* (high avg_logprob on repetition loops), so "low-confidence→escalate" systematically misses the catastrophic cases → worse-than-random. Quantify the effect and confirm it's a real failure mode, not a bug.

**Deliverables.** Hallucination-trigger curves vs the champion scalar; confidence-autopsy figures; a "does hallucination-flag capture the Indian-English gain?" slice analysis; report; COMPARISON rows.

**Acceptance.** Hallucination triggers scored on the shared axes; confidence sign verified and the −1.33 explained.

**Decision rule (pre-registered).**
- A trivial hallucination trigger **captures most of the Indian/US gain** → a genuine threat to the scalar's marginal value; surface it honestly and **reframe the scalar's contribution** as the *within-accent, non-hallucination* gradient it adds beyond the flag. Test a composed router (scalar ∪ hallucination-flag) as the practical champion.
- Hallucination trigger **does not capture the gain** → the scalar's edge is more defensible; report that the benefit is not reducible to hallucination detection. Keep the flags as backstops in the reference set.

---

### Extension 4 — Statistics pooling (mean + std over time) (EXP-07) → Gate 4  *(pause before starting)*

**Question.** Does preserving *within-utterance* variation (std over time) catch the local events (mumbles, pauses, hallucination triggers) that mean-pooling dilutes?

**Pre-start gate.** This is the only compute-heavy step this round: re-extract WavLM saving **per-layer std alongside mean** (→ effectively `(25, 2048)`; ~2× storage, CPU minutes–~1h over 900 utts). **Do not save the full temporal tensor (~13 GB).** Confirm at Gate 3's close before running the re-extraction.

**Method.** Extend the feature extractor to emit mean **and** std per layer (NaN guard preserved). Retrain the champion probe on concatenated mean+std. Ablate **mean-only vs std-only vs mean+std**. Focus the analysis on the catastrophic/high-misrouting-cost subset — does std specifically help there?

**Deliverables.** New feature cache (std); three-way pooling ablation on the frozen test fold; subset analysis; report; COMPARISON rows.

**Acceptance.** mean+std probe scored on shared axes with CIs; ablation reported.

**Decision rule (pre-registered).**
- Improvement **concentrated on the high-misrouting-cost subset** → temporal-locality hypothesis supported; std pooling is a cheap real win.
- **Flat** → confirms the working assumption that time-pooling is a minor lever here; report as a clean negative result and stop investing in pooling.

---

### Extension 5 — Multi-task (gain/WER + accent) probe (EXP-08) → Gate 5  *(optional/stretch)*

**Question.** Does accent supervision help the router, or does it just collapse the scalar back toward accent identity (i.e. reproduce argmax)?

**Method (from cache).** Shared trunk, two heads: regression (champion target) + accent classification. Loss = reg + λ·CE, sweep λ ∈ {0, 0.1, 0.3, 1.0} (λ=0 recovers Ext 1). Report, for each λ: net-WER curve, layer specialization, and **MI(score, accent)** and **MI(score, speaker)** as in the pilot leakage analysis.

**Deliverables.** λ-sweep curves; MI-vs-λ plot; layer-weight-vs-λ; report; COMPARISON rows.

**Acceptance.** λ-sweep scored on shared axes; MI reported.

**Decision rule (pre-registered).**
- λ>0 **improves net WER without materially raising MI-to-accent** → genuine auxiliary benefit; adopt.
- λ>0 helps **only by raising MI-to-accent** (the scalar is encoding accent identity) → it's argmax in disguise; report as such and do not adopt. This is the outcome I expect; proving it cleanly is a good interview result.

---

## 5. Synthesis & write-up (→ Final gate)

- Regenerate the master overlay (`make compare`) and finalize `experiments/COMPARISON.md` as the round's headline artifact: every trigger and extension on the frozen test fold under corrected metrics, with CIs.
- Write **`writeup/writeup-v2.md`** (keep `writeup.md` as the pilot record): fold in the diagnosis, the corrected metrics (capped WER, AUC/AP, CIs — and whether the scalar's edge over argmax is significant), and each extension's verdict via its decision rule. Keep negative/conditional results first-class. Add a short "pilot → v2" narrative so the progression is legible.
- Update `docs/DECISIONS.md` and `docs/ai-use-log.md`.
- Ensure `make reproduce-v2` runs the whole second round from cache (Phase 0 → available extensions → compare) from a clean checkout.

---

## 6. Repo additions

- **Docs:** `docs/DIAGNOSIS-pilot-to-v2.md`.
- **Code:** `src/accentedness_routing/eval/eval_common.py` (canonical scorer); extend `features/` for std pooling (Ext 4); extend `triggers/` for hallucination flags (Ext 3) and the multi-task head (Ext 5). Do **not** fork the routing/curve logic anywhere.
- **Experiments:** `EXP-04…EXP-08` folders; `COMPARISON.md`; updated `INDEX.md`; `figures/operating_curves_all.png`.
- **Make targets:** `make diagnose` (Phase 0), `make ext1 … make ext5`, `make compare`, `make reproduce-v2`. Each `extN` target imports `eval_common`; none re-runs ASR except `ext4`'s re-extraction step (guarded).
- **Write-up:** `writeup/writeup-v2.md`.

---

## 7. Experiment reporting protocol (extended)

Each `EXP-<NN>-<slug>/report.md` contains, in this order: **date + commit hash**; **question**; **config** (models, subset, seed, target, threshold sweep, lib versions); **method** (one paragraph); **results** (curve figure + metrics table with **bootstrap CIs**, netWER@budgets, AUC/AP, per-accent/per-L1 deltas); **the filled-in Decision rule** (which branch fired, with the numbers that decided it); **caveats / negative findings** (stated plainly); **what this changes** (champion update, next action). Save `metrics.json` and figures alongside. Append one row to `COMPARISON.md` and update `INDEX.md`. Reports must be skimmable.

---

## 8. What to present at each gate

Keep each gate report to a few lines so I can approve fast:

1. **Which decision-rule branch fired**, with the two or three numbers that decided it (and their CIs).
2. **The updated `COMPARISON.md` row(s)** for this extension against the reference set.
3. **Recommended next action** — proceed to the next extension as planned, or a proposed deviation (with the reason) parked in `OPEN-QUESTIONS.md`.
4. Anything that would change my priorities (e.g. Ext 2 says data-limited → CV pull recommended; Ext 3 says hallucination-flag eats the gain → scalar reframed).

---

## 9. Open knobs to confirm in your plan (don't block — propose defaults)

1. **τ for the gain>τ scorecard** (default: report both τ=0.0 and τ=0.05).
2. **Bootstrap draws** (default 1000; bump to 2000 if fast enough).
3. **Champion-probe carry-forward policy** (default: the best target from Ext 1 becomes the trunk for Ext 2/4/5; state it explicitly once Ext 1 lands).
4. **Hallucination thresholds** (default: sweep compression_ratio and no_speech_prob rather than hard-coding Whisper's 2.4 / 0.6, and report the swept curve).

Produce the Round-2 plan now (task breakdown + new files/targets + the four defaults above). Choose sensible values for the defaults and state them. Wait for my approval (**Gate 0**) before implementing; after that, proceed autonomously *within* each phase but **stop at Gate A and after every extension** per §0.1.
