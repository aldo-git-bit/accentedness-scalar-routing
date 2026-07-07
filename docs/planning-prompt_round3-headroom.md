# Claude Code Planning Prompt — Round 3: Finding the Regime Where Trigger Quality Matters

> **How to use this prompt.** Paste into Claude Code in the existing `accentedness-scalar-routing` repo. **First produce a written implementation plan** (phase-by-phase breakdown + new files/targets) and stop for my review (**Gate 0**). Do not code until I approve.
>
> After approval, work autonomously *within* a phase; commit and write an experiment report after each. **Every phase is designed to leave the project writeup-ready if I stop there** — so structure each experiment so its report stands alone and the synthesis can be written from whatever subset completed. Pause only at the hard gates in §0.1. Don't over-engineer the gate mechanics; the point is that a partial run is always a coherent deliverable.

---

## 0. Context — what Round 2 established, and the reframing that drives Round 3

Round 2 corrected the pilot's contaminated metric (capped WER, AUC/AP, bootstrap CIs on a frozen 179-utt test fold) and ran a five-way ablation. The decisive result is not about any single trigger — it is about **headroom**:

- **Oracle area-vs-random is only 0.0311.** That is the entire routing prize on this data/model pair.
- **Confidence (Whisper `avg_logprob`) captures 0.0147 — ~47% of the oracle prize — for free, no training.** It is the best non-oracle trigger and its CI excludes zero.
- The best *learned* trigger (multitask λ=0.1) reaches 0.0108 but **does not significantly beat confidence** (paired CI includes zero).
- Median capped escalation gain is **0.0 for every accent**; only **47/179** utterances have gain > 0.05. The signal is sparse and concentrated in 2–3 accents (Indian English mean gain 0.123; US English slightly *negative*).
- The learning curve is **flat/declining** past 75% of train data → **not data-limited for this target**. The binding constraint is **signal sparsity in the target**, which is downstream of **headroom**: small→large-v3 simply don't differ on most utterances.

**The reframing for Round 3.** The residual room any trigger could win over confidence is ≈ 0.031 − 0.0147 ≈ **0.016**, and paired-bootstrap CIs are ≈ ±0.012 wide at n=179. **The current setup cannot statistically distinguish a good learned trigger from confidence — there is barely any routing value to capture and not enough test power to resolve it.** Chasing a "better scalar" inside this regime will keep returning CIs that include zero.

So Round 3 stops trying to build a better trigger inside a fixed problem and instead **manipulates the problem to find the regime where trigger quality becomes detectable at all** — chiefly by widening the model gap (creating headroom), and by testing whether *anything* — composite signals, temporal features, or an accent-*adapted* careful path — beats free confidence once there is room to. This is the honest resolution of the project's thesis and is directly the "when do you actually need accent-aware routing" question. A clean negative ("free confidence is the right production answer until the model gap is wide") is a first-class result.

Framing/scope docs (`docs/what-is-accent.md`, `accent-routing-sota.md`, `accent-routing-scoping.md`, `six-proposals.md`) remain the source of truth. Non-negotiables from Round 2 carry over verbatim: **frozen test fold; one shared `eval_common.py`; capped WER everywhere; curves-with-CI not points; trigger from audio/SSL/metadata never transcript; pretrained-first (only the small probe/combiner trains).**

### 0.1 Hard gates (pause and wait)

| Gate | When | Why |
|---|---|---|
| **Gate 0** | After the plan, before code | Approve the plan |
| **Gate A** | End of Phase 0 | Approve the diagnosis addendum + grid-aware harness before spending decode budget |
| **Gate S** | *Before* the WavLM re-extraction in Phase 3 | ~1 hr CPU compute — confirm before running |
| **Gate D** | *Before* the model download in Phase 4 | Downloads an unverified HF finetune; confirm |

Within a phase, default to action, log defaults in `docs/DECISIONS.md`, park non-blocking questions in `docs/OPEN-QUESTIONS.md`, keep `docs/ai-use-log.md` current. Phases run in priority order (1→4); **do not reorder or skip ahead.** I expect to reach Phases 1–2 for sure; 3–4 are stretch.

---

## 1. Objective

Determine **under what conditions, if any, a learned or composite trigger beats free confidence** — by creating headroom and testing each candidate mechanism against the confidence bar. Concretely:

1. **Headroom grid** (Phase 1): sweep the model gap and re-run the full comparison per pairing; produce the headline **headroom-sweep figure**.
2. **Composite trigger** (Phase 2): test whether confidence *plus* cheap learned/acoustic signals beats confidence *alone*, under narrow and wide gaps.
3. **Targeted temporal features** (Phase 3): test whether per-layer temporal **std** catches the *confidently-hallucinated* cases confidence provably routes wrong (Round 2: 56% of hallucinations have above-median `avg_logprob`).
4. **Accent-adapted careful path** (Phase 4, gated stretch): test "try *differently*, not just harder" on the Indian-English slice — does an accent-adapted careful model expand headroom in a way a bigger *general* model cannot?

Each phase carries a **pre-registered decision rule** mapping outcomes (including nulls) to conclusions.

---

## 2. Non-negotiable comparability rules (carried from Round 2, extended for the grid)

1. **Frozen test fold** — identical speaker-disjoint 179-utt test fold; never re-split. Combiner/probe training uses the existing train/val folds only; subsample **by speaker**.
2. **One shared scorer** — everything routes through `eval_common.py`. Extend it for the grid; do not fork curve/metric logic.
3. **Capped WER everywhere**; report **micro and macro**.
4. **Curves with bootstrap 95% CI** (≥1000 resamples); pairwise claims use **paired bootstrap on the difference**, never two overlapping bands. The bar to beat is **confidence**, so every "does X help" claim is a paired bootstrap **vs confidence** (and, where relevant, vs confidence-composite).
5. **Reference set carried forward** in every figure/table: oracle, random, confidence, argmax-accent, and the Round-2 champion (multitask λ=0.1).
6. **Grid-awareness (new):** every trigger's score is computed **per (default, careful) pairing**. Note two triggers are **default-specific** — `avg_logprob` (confidence) and `no_speech_prob` come from the *default* model's decode, so each pairing has its *own* confidence trigger. The learned probe's target (capped gain) is pairing-specific, so **retrain the champion recipe per pairing** (seconds on cached WavLM features).
7. **Determinism**; log lib versions per report.

---

## 3. Phase 0 — Diagnosis addendum + grid-aware harness (→ Gate A)

No new modeling; record the reframing and prepare the harness.

**3.1 `docs/DIAGNOSIS-round2-to-round3.md`** — the transition record. Must argue, in prose: headroom is the binding constraint (oracle 0.031; confidence captures 47%; residual ≈0.016 < CI width ≈±0.012 → the setup is **underpowered to distinguish a good trigger from confidence**); therefore Round 3 manipulates headroom rather than the trigger; and a negative result ("use free confidence until the gap widens") is a legitimate, product-relevant finding. State the three Round-3 levers (widen gap; composite; accent-adapted path) and what each would prove.

**3.2 Extend `eval_common.py`:**
- Parameterize all scoring by `(default_model, careful_model)` pairing.
- Add a **headroom summary** per pairing: oracle area-vs-random, plus the **WER gap** = mean capped default WER − mean capped careful WER, and the count of utterances with capped gain > {0, 0.05}.
- Add a **combiner-eval** entry point (Phase 2): fit on val, score on test, paired-bootstrap vs confidence.
- Unit tests: default-specific triggers pull from the correct model's metadata; grid loops don't leak across pairings.

**3.3 Comparison backbone:** extend `experiments/COMPARISON.md` to a grid schema (rows = trigger × pairing). Extend `make compare` to emit the **headline headroom-sweep figure** (below). Update `INDEX.md`.

**Gate A.** Present the diagnosis addendum + the grid schema + a stub of the headroom figure axes. **Stop.**

---

## 4. Phase 1 — Headroom grid (EXP-09) → the headline deliverable

**Question.** As the model gap widens, does headroom grow, does confidence keep capturing most of it, and does any learned trigger separate from confidence?

**Design.** Grid of **default ∈ {tiny, base, small}** × **careful ∈ {turbo, large-v3}** = 6 pairings (small→large-v3 is the Round-2 baseline, already cached).

**ASR (three new decodes; small + large-v3 cached).** Add `tiny`, `base`, and `large-v3-turbo` via mlx-whisper; cache per-utterance hypothesis, capped WER, `avg_logprob`, `no_speech_prob`, wall-clock, keyed by (utterance, model), exactly as Round 2. Confirm exact mlx-community repo IDs (Round-2 cache used the `mlx-community/whisper-<size>-mlx` naming, e.g. `...whisper-small-mlx`, `...whisper-large-v3-mlx`; verify `whisper-tiny-mlx`, `whisper-base-mlx`, `whisper-large-v3-turbo` exist under `mlx-community` and adjust if the turbo repo differs). WavLM features are **default-agnostic → reuse the existing cache**.

**Per pairing, compute the full comparison** through `eval_common`: oracle, random, that pairing's **confidence** and **no_speech_prob**, argmax-accent, and the **retrained champion probe** (target = that pairing's capped gain). All with CIs; paired-bootstrap each learned/heuristic trigger **vs that pairing's confidence**.

**Headline figure (`experiments/figures/headroom_sweep.png`).** x-axis = WER gap (or oracle area) per pairing; y-axis = area-vs-random; one series each for **oracle**, **confidence**, and **best-learned-trigger**; encode **careful-path cost** (turbo vs large-v3) by marker/color. This one plot is the round's primary artifact.

**Deliverables.** New ASR caches; per-pairing COMPARISON rows; headroom figure; report.

**Decision rule (pre-registered).**
- **Oracle area grows monotonically with the gap** → headroom is manufacturable (expected); the sparsity ceiling was a property of small→large-v3, not of the task.
- **Confidence keeps capturing ~most of the headroom at every gap** → "free confidence is the production answer regardless of gap"; learned routing never earns its keep here. Strong, clean finding.
- **A learned trigger separates from confidence (paired CI excludes zero) only past some gap** → you've *located the regime* where the scalar earns its keep — the project's thesis, resolved. Carry that pairing forward as the primary setting for Phases 2–4.
- **Turbo captures ≈ large-v3's headroom at lower cost** → report the selective-spend implication (cheap careful path suffices), independent of trigger quality.

---

## 5. Phase 2 — Composite trigger: does anything beat confidence? (EXP-10)

**Question.** Nobody has tested confidence *combined* with other signals — only against them. Does a small combiner over cheap signals beat confidence alone?

**Method.** Fit a **regularized combiner** (L2 logistic regression or a shallow GBM; pick by val, log the choice) on the **val fold**, target = capped gain > τ, over already-computed / near-free features: **confidence, no_speech_prob, the champion learned score, and cheap acoustic features** (utterance duration, silence ratio, speaking rate — extractable from audio/VAD, no re-decode). Evaluate on the frozen test fold; **paired bootstrap vs confidence-alone**. Run under **two regimes from EXP-09**: the narrow gap (small→large-v3) and the widest gap (tiny→large-v3).

**Power discipline (bake in).** With 179 test utts this is power-limited: **select on val, confirm on test, report the null honestly**; no threshold-hunting on test. State the paired-difference CI regardless of sign.

**Decision rule.**
- Composite beats confidence (paired CI excludes zero) at the wide gap but not narrow → confirms Phase 1's regime story from a second angle; the learned signal is *additive* to confidence where headroom exists.
- Composite ≈ confidence everywhere → the learned/acoustic signals are **subsumed by confidence**; report that free confidence is sufficient. Legitimate, informative null.

---

## 6. Phase 3 — Temporal std as a targeted attack on confidence's blind spot (EXP-11) → Gate S

**Question.** Round 2's sharpest finding: **Whisper hallucinates confidently** (56% of hallucinations above median `avg_logprob`) — those are exactly the utterances confidence routes *wrong*. Does per-layer temporal **std** detect the confidently-hallucinated cases confidence misses? (This also finally runs the deferred EXP-07 stats-pooling infra, closing the one untested branch of the three-way diagnosis.)

**Gate S:** the WavLM re-extraction (mean **and** std per layer → `(25, 2048)`, ~2× storage, ~1 hr CPU over 900 utts; **not** the 13 GB temporal tensor). Confirm before running.

**Method.** Re-extract with std; fold std features into the Phase-2 combiner. Evaluate specifically on the **high-misrouting-cost / confidently-hallucinated subset** (utterances where confidence routes wrong), not just the aggregate. Ablate mean-only vs mean+std within the combiner.

**Decision rule.**
- Improvement **concentrated on the confidently-hallucinated subset** → temporal info matters precisely for confidence's blind spot; a real, well-motivated win, and the architecture branch resolves *positive*.
- **Flat** → confirms mean-pooling wasn't the binding constraint; the architecture branch of the diagnosis resolves *negative* and is now fully tested. Clean close.

---

## 7. Phase 4 — Accent-adapted careful path: "try differently, not just harder" (EXP-12) → Gate D (stretch)

**Question.** Every prior path routes to a bigger *general* model, which can only try harder. Does an accent-*adapted* careful model expand headroom **accent-specifically** on the Indian-English slice (Round-2 mean gain 0.123) in a way large-v3-general cannot?

**Gate D + reliability guard (critical — the landscape is a minefield).** The **only** plausibly-correct candidate is **`Tejveer12/Indian-Accent-English-Whisper-Finetuned`** (a `large-v3-turbo` finetune on an Indian-English-accent set; HF PyTorch, MIT-ish — verify license). **Do NOT use** `Oriserve/Whisper-Hindi2Hinglish-*`, `vasista22/whisper-hindi-*`, or `parthiv11/indic_whisper_*` — these are Hindi / Hinglish / Indian-*languages* models and will produce garbage against EdAcc's English references. **Infra wrinkle:** it's PyTorch, not MLX — either convert with `mlx_whisper.convert` or run via `transformers` on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`); pick the simpler working path and log it.

**Verify-before-trust (do this first, and stop if it fails).** Run the candidate on the Indian-English test slice and compare WER to **large-v3-general** and **turbo-general** on the *same slice*. If it does **not** clearly beat general large-v3 on Indian English, **report the negative result and stop** — do not build routing on an unreliable specialist. Log the check in DECISIONS.

**Method (only if the guard passes).** Define the careful path for the Indian slice as the adapted model; compare headroom (oracle area, and confidence/champion-triggered net WER) for {general large-v3 careful} vs {accent-adapted careful} on the Indian slice. The question is whether "try differently" opens headroom that "try harder" doesn't.

**Decision rule.**
- Adapted path expands Indian-slice headroom beyond general large-v3 → concrete evidence for accent-*aware* routing over pure quality escalation; the strongest on-thesis result and a headline "when you actually need accent-awareness" finding.
- Guard fails or headroom doesn't expand → report plainly; the general careful path is sufficient, and this becomes the "with another week / better specialist" argument.

---

## 8. Phase 5 — Synthesis + `writeup/writeup-v3.md`

- Regenerate `make compare` and the headroom figure as the round's headline.
- Write `writeup-v3.md` (keep v2): lead with the headroom reframing and the headroom-sweep result; then composite, temporal, and (if run) accent-adapted findings via their decision rules; keep nulls first-class. Add a short "pilot → v2 → v3" arc so the progression reads as a coherent scientific narrative (contamination fixed → headroom identified as binding → regime-finding).
- Update `DECISIONS.md`, `ai-use-log.md`. Ensure `make reproduce-v3` runs the whole round from cache (re-extraction and the download stay gated/optional).
- **Deferred to a possible Round 4 (do not start):** Common Voice scale-up — only pays off *combined* with a wide gap (Round 2 showed no data-limitation for the current target), and it's download-gated. Note it as the next lever on test power, not a Round-3 task.

---

## 9. Repo additions

- **Docs:** `docs/DIAGNOSIS-round2-to-round3.md`.
- **Code:** extend `eval_common.py` (grid + combiner eval); `asr/` decode wrappers already exist — add tiny/base/turbo model configs; extend `features/` to run the built-but-unrun std extraction (EXP-07 infra); add a small `triggers/combiner.py`; `asr/` path for the adapted model (convert-or-MPS).
- **Experiments:** `EXP-09-headroom-grid`, `EXP-10-composite`, `EXP-11-temporal-std`, `EXP-12-accent-adapted`; grid-schema `COMPARISON.md`; `figures/headroom_sweep.png`; updated `INDEX.md`.
- **Make targets:** `make asr-ladder` (tiny/base/turbo decodes), `make ext-headroom`, `make ext-composite`, `make features-stats && make ext-temporal`, `make ext-adapted`, `make compare`, `make reproduce-v3`. Only `asr-ladder` (3 fast decodes), `features-stats` (Gate S), and `ext-adapted` (Gate D) cost anything new.
- **Write-up:** `writeup/writeup-v3.md`.

---

## 10. Experiment reporting protocol

Per `EXP-<NN>-<slug>/report.md`: date + commit; question; config (models, pairing(s), seed, features, combiner choice, lib versions); method (one paragraph); results (headroom/curve figure + metrics table with **bootstrap CIs**, paired-bootstrap **vs confidence**, per-accent/L1 where relevant); **filled-in decision rule** (which branch fired, with the numbers); caveats / nulls stated plainly (esp. power limits at n=179); what this changes / next action. Save `metrics.json` + figures; append COMPARISON rows; update `INDEX.md`. Each report must stand alone so a stop after any phase is writeup-ready.

---

## 11. Open knobs to confirm in your plan (don't block — propose defaults)

1. **Grid extent** (default: {tiny, base, small} × {turbo, large-v3} = 6; small→large-v3 reuses cache).
2. **Combiner family** (default: L2 logistic; try shallow GBM if it clears logistic on val; report choice).
3. **Champion recipe carried into the grid** (default: multitask λ=0.1, the best Round-2 learned trigger; state it once EXP-09 confirms).
4. **Acoustic features for the composite** (default: duration, silence ratio, speaking rate — cheap, no re-decode).

Produce the Round-3 plan now (task breakdown + new files/targets + the four defaults). Choose sensible defaults and state them. Wait for my approval (**Gate 0**); then proceed autonomously within each phase, pausing only at Gates A / S / D, and keeping every phase writeup-ready.
