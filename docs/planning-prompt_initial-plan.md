# Claude Code Planning Prompt — Accentedness-Scalar Routing for English ASR

> **How to use this prompt.** Paste this into Claude Code. **First produce a written implementation plan** (a phase-by-phase task breakdown plus the proposed repo tree) and stop for my review. **Do not start coding until I approve the plan.** Once I approve, **execute all phases autonomously, end to end, without waiting for further check-ins** — I may be away from the desk, so keep making progress rather than pausing for confirmation (see §0.1). Commit after each phase and write an experiment report after every experiment (see §8). You are pre-authorized to download the required assets (EdAcc ~7 GB and the specified Whisper / WavLM / CommonAccent checkpoints, plus a small optional CV/LibriSpeech subset). Pause and wait for me **only** for the hard gates in §0.1.

---

## 0. Roles, ownership, and handoff

You are implementing a self-contained research prototype end to end on my machine. I am the repo owner. This is a take-home exercise for a **Staff/Research Data Science interview at Deepgram**. When it is done I will push it to a **private GitHub repo under `https://github.com/aldo-git-bit`** and share it with the interviewers, who will **clone it, read it, and run it to evaluate my work**. Therefore:

- The repo must be **clean, reproducible, and self-explanatory to a stranger**. Assume the reader is a skeptical expert who will run it.
- The **README must contain complete run instructions** (the assignment explicitly requires "any instructions needed for us to run the work").
- Favor **judgment, honesty, and defensibility over scale.** Negative and conditional results are valued; hiding them is not. The interview spends its first third on what I built, so every artifact must be defensible under questioning.
- Keep a running **`docs/DECISIONS.md`** (design decisions + rationale) and **`docs/ai-use-log.md`** (what was delegated to AI, accepted, corrected, rejected, and how it was validated) as you go — these feed the final write-up and satisfy the assignment's AI-use note.

### 0.1 Working autonomously (I may be away from the desk)

After I approve the plan, **do not wait for me between phases.** Default to action: make progress continuously, and when you hit a decision that isn't a hard gate below, **choose the sensible default, log it in `docs/DECISIONS.md`, and keep going** — do not stall. Park any non-blocking questions in `docs/OPEN-QUESTIONS.md` (and in the relevant experiment report) for me to review asynchronously; batch them rather than interrupting.

**You can and should do without my input:** repo scaffolding; environment setup; data download / prep / subsetting / speaker-held-out splits; ASR + WER runs and caching; feature extraction; all baselines; probe training and threshold calibration; evaluation, slicing, and plots; writing experiment reports; updating README / docs / DECISIONS / ai-use-log; committing and pushing; running and fixing smoke and unit tests; debugging your own errors and retrying; choosing the four defaults in §10; and drafting the write-up. If a phase's acceptance check fails, **diagnose and fix it yourself** before moving on.

**Pause and wait for me only if:** (a) a download would exceed ~15 GB or pull an asset not listed in this prompt; (b) something would require training heavier than the small probe (e.g. fine-tuning an ASR model — this should never be needed here); (c) you become convinced the §3 design is wrong and want to deviate materially — in that case, implement §3 as specified, record the concern in `docs/OPEN-QUESTIONS.md`, and continue; or (d) you are truly blocked by missing credentials/access or an unrecoverable error. Even when blocked, **keep working on any independent parts of the plan** while the blocker waits for me.

---

## 1. Research Objective

This project builds and evaluates an **accentedness-scalar router** for English automatic speech recognition. Rather than classifying a speaker's accent into a discrete category, it reads a continuous accentedness/difficulty signal from self-supervised speech representations and uses it to decide, per utterance, whether to escalate from a fast default recognizer to a more careful one. Success is measured not as classification accuracy but as **net word-error-rate reduction against escalation cost** — a quality-versus-cost operating curve — evaluated on conversational, accent-labeled speech, and compared against a naive discrete-accent-classifier router, with gains expected to concentrate on gradiently-accented and historically under-served speakers.

## 2. Context

Accent is not a discrete label but a **gradient, feature-specific** property: a speaker may carry native vowels with second-language prosody, or shade from lightly to heavily accented along a continuum. ASR systems degrade on accented speech not because those speakers are deficient but because the model is **miscalibrated** to varieties under-represented in its training data. This reframes the task: an accent signal has little intrinsic value; what matters is the **downstream effect of acting on it** — detection is an instrument, not the product. A discrete argmax classifier forces hard categories onto a continuous phenomenon and cannot express "somewhat accented," which is precisely the region where routing decisions are most delicate. A continuous scalar matches the gradient nature of accent and — because it measures *degree of difficulty* rather than identity — doubles as a general quality signal for the recognizer.

A secondary motivation is the **unbounded-taxonomy problem**: discrete accent inventories break under distribution shift, because real deployments field far more accents than any fixed label set can enumerate, and a scalar's label-free nature absorbs unseen accents as a side benefit rather than by design. This is a *supporting* argument, not the core one — the primary case for the scalar is gradience and the instrument framing above. Finally, routing exists to **spend selectively**: escalating every utterance is just "use the bigger model" and destroys the rationale. Success is therefore intrinsically a trade-off — net WER measured against escalation rate and added latency — reported as an **operating curve** rather than a single number.

*(The `docs/` knowledge base — `what-is-accent.md`, `accent-routing-sota.md`, `accent-routing-scoping.md`, `six-proposals.md` — contains the full argument and citations behind this framing. Read them before designing; they are the source of truth for scope.)*

---

## 3. Technical Design (the spine — implement exactly this unless we agree to change it)

**The routing decision.** Per utterance, choose **default path** (a fast/cheap Whisper) vs **careful path** (a larger Whisper). The "improvement" from routing is escalating the hard utterances to the careful model while leaving the rest cheap.

**The trigger under test (the contribution).** A **learned accentedness/difficulty scalar**: extract WavLM-large hidden states, take a **learnable softmax-weighted sum across all layers** (SUPERB-style), **mean-pool over time** to one vector per utterance, and train a **small probe** to predict the utterance's **base-model WER** (a difficulty/accentedness proxy). Threshold the predicted scalar to route. Train the probe **speaker-held-out**.

**What we compare it against (same score→curve protocol for all):**
- **No-routing floors:** default-always and careful-always (bracket the curve).
- **Oracle router:** escalate exactly the utterances where the careful model actually helps (upper bound).
- **Random router:** escalate a random X% (lower bound / sanity).
- **Argmax-classifier baseline (the key baseline P2 must beat):** the CommonAccent XLSR classifier's signal (e.g. `1 − max_softmax`, or a "not a well-served accent" rule) as a routing score.
- **Confidence baseline (P1-style, for context):** Whisper's own confidence (avg logprob / no-speech prob) as a routing score.

**The headline metric.** A **quality-vs-cost operating curve**: net WER (careful where escalated, default elsewhere, misrouting cost included) vs **escalation rate**, plus **added latency**. Summarize each trigger by (a) net WER at a fixed escalation budget (e.g. 20%) and (b) area between its curve and the random baseline. **P2 wins if the scalar curve dominates the argmax-classifier curve**, especially on accented/under-served slices.

**Mandatory slicing & diagnostics.** Report net-WER delta **per accent** and **per L1** (EdAcc provides both); report the **substitution/deletion** profile; verify the scalar is **not just speaker identity** (speaker-held-out splits are the guard; also report scalar↔accent vs scalar↔speaker association). Honest **per-slice / conditional / negative** findings ("helps Indian English, hurts X") are first-class results.

**Non-negotiable constraints (from the scoping doc):**
- Trigger is read from **audio/SSL**, never from decoded transcript text.
- **Pretrained-first**; the only training is the small probe (+ learnable layer weights) and threshold calibration. No fine-tuning of ASR, no training from scratch.
- Report a **curve, not a point**; never WER without a cost axis.

---

## 4. Environment & Infra (local, Apple Silicon)

Target: **Apple Silicon M2/M4, ~24 GB unified memory, local-first.** Everything fits locally; no cloud needed.

- **Python 3.10–3.12** (avoid 3.13 — several speech deps lack wheels). Use a **venv** (or `uv`); pin deps in `pyproject.toml` / `requirements.txt`.
- **Two runtimes, no MLX porting work:**
  - **ASR leg → MLX:** `mlx-whisper` (turnkey). Default cheap path `mlx-community/whisper-small`; careful path `mlx-community/whisper-large-v3` (turbo optional). Models auto-download to `~/.cache/huggingface`.
  - **Feature/classifier leg → PyTorch:** WavLM-large + CommonAccent XLSR. **Run WavLM feature extraction on CPU** at this scale (minutes; avoids MPS op-coverage flakiness). MPS optional with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Known gotcha — WavLM NaNs:** load with `WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)`, **not** `device_map=`; add a smoke test asserting hidden states are finite (transformers issue #31970).
- **Audio:** EdAcc is **32 kHz** — resample to **16 kHz** (`datasets` `cast_column("audio", Audio(sampling_rate=16000))`) since WavLM and Whisper expect 16 kHz. `ffmpeg` required (`brew install ffmpeg`).
- **Secrets:** HF token via `huggingface-cli login`; Common Voice/MDC API key in a **`.env`** that is **gitignored**. Never commit tokens.
- **Core libs:** `mlx-whisper transformers datasets torch torchaudio speechbrain jiwer scikit-learn librosa soundfile numpy pandas matplotlib`.

---

## 5. Data

**Primary (required): EdAcc** — `edinburghcstr/edacc`, public/ungated, CC-BY-SA, ~7 GB. Conversational, accent-labeled English. Fields: `speaker`, `text`, `accent` (linguist-standardized), `raw_accent` (self-reported), `l1`, `gender`, `audio`. **Splits are validation + test only (no train split).**
- Build **speaker-disjoint** folds: pool validation+test, then split into probe-train / probe-val / test folds with **no speaker overlap**, stratified by accent where feasible, fixed seed. Document the split.
- Subset for speed: cap utterances per accent and pick a manageable accent set; keep it configurable. Start small.
- Treat `raw_accent` (self-report) as a **noisy proxy**, `accent` (standardized) as the working label.

**Optional read-speech contrast (exposes read-vs-conversational domain shift):**
- **Common Voice** via Mozilla Data Collective (I have an MDC account + API key; loader in `.env`). Use the `datacollective-python` library; pull a **small English subset** (a few hundred–thousand clips) — do **not** download all of English. CV's `.tsv` carries self-reported `accents`.
- **Fallback if MDC is slow:** **LibriSpeech** (`openslr/librispeech_asr`, ungated) as a clean read-speech floor (no accent labels).
- **This contrast is secondary — never let it block the EdAcc core.**

**Text normalization:** apply a consistent normalizer (Whisper `EnglishTextNormalizer` or jiwer transforms — lowercase, strip punctuation, etc.) before all WER computation.

---

## 6. Repo Structure

Repo name: **`accentedness-scalar-routing`**. Data and model weights are **gitignored**.

```
accentedness-scalar-routing/
├── README.md                  # what it is, install, quickstart, full reproduction, results summary, run instructions
├── pyproject.toml             # pinned deps, Python 3.10–3.12
├── requirements.txt
├── .gitignore                 # data/, models/, .env, caches, __pycache__, *.wav
├── .env.example               # names of required env vars (HF token, MDC key) — no values
├── Makefile                   # make setup / smoke / features / baselines / probe / eval / report / reproduce
├── docs/                      # planning + knowledge base (I will drop these in)
│   ├── what-is-accent.md
│   ├── accent-routing-sota.md
│   ├── accent-routing-scoping.md
│   ├── six-proposals.md
│   ├── research-objective.md  # §1–§2 of this prompt, verbatim
│   ├── DECISIONS.md           # running design decisions + rationale
│   └── ai-use-log.md          # delegated / accepted / corrected / rejected / validated
├── src/accentedness_routing/
│   ├── data/                  # EdAcc load, subset, 16k resample, speaker-held-out splits; optional CV/LibriSpeech
│   ├── asr/                   # mlx-whisper wrappers (default + careful), per-utterance WER (jiwer), caching
│   ├── features/              # WavLM hidden-states, learnable weighted-sum, mean-pool, NaN guard, caching
│   ├── triggers/              # scalar probe; argmax-classifier baseline; confidence baseline
│   ├── routing/               # score→route, escalation, net-WER (misrouting included), latency accounting
│   ├── eval/                  # curves, per-accent/L1 slicing, diagnostics, plots
│   └── flywheel/              # (optional) drift detection + hard-case mining
├── scripts/                   # thin CLIs: prepare_data, extract_features, run_baselines, train_probe, evaluate, make_report
├── configs/                   # yaml experiment configs (models, subset sizes, thresholds, seeds)
├── experiments/               # one dated folder per experiment (see §Experiment Reporting)
├── tests/                     # smoke tests: WavLM finite-hidden-states, tiny end-to-end, WER sanity
├── notebooks/                 # optional: one demo notebook showing baseline vs improvement
├── writeup/                   # 2–3 page write-up + AI-use note (final deliverable)
├── data/                      # gitignored
└── models/                    # gitignored (downloaded ckpts, trained probe.pt)
```

---

## 7. Phased Implementation Plan

Work top to bottom. Each phase: **goal → tasks → deliverables → acceptance check.** Commit at the end of each phase. **Gate the whole pipeline behind a tiny-subset smoke run (Phase 4.5) before any full run.**

**Phase 0 — Environment & preflight.** venv, pinned deps, `huggingface-cli login`, `ffmpeg`, `.env` from `.env.example`. Smoke-test: load WavLM-large and assert hidden states are finite; transcribe one EdAcc clip with mlx-whisper; compute one WER with jiwer. *Acceptance:* all three smoke checks pass.

**Phase 1 — Repo scaffold + docs + git.** Create the tree in §6, populate README skeleton, drop §1–§2 into `docs/research-objective.md`, init git, add `.gitignore`/`.env.example`, create the private remote under `aldo-git-bit`, first commit. *(I will place the four knowledge-base docs into `docs/`.)* *Acceptance:* repo pushes privately; `docs/` present.

**Phase 2 — Data pipeline.** EdAcc loader with 16 kHz resample, subset config, and **speaker-disjoint** train/val/test folds (fixed seed, accent-stratified where feasible). Persist the split manifest. Optional CV/LibriSpeech read-contrast loader (clearly marked optional). *Acceptance:* split manifest has zero speaker overlap across folds; per-accent counts printed.

**Phase 3 — ASR + WER harness.** mlx-whisper wrappers for default and careful models; per-utterance normalized WER via jiwer; **cache** transcripts + per-utterance WER to disk (keyed by utterance id + model). Record per-utterance wall-clock for latency accounting. *Acceptance:* cached WER tables for both models over the subset; re-runs hit cache.

**Phase 4 — Feature extraction.** WavLM-large hidden states → learnable weighted-sum (init uniform) → mean-pool → per-utterance vector; cache vectors. NaN guard enforced. *Acceptance:* feature cache built; no NaNs; vector dim == 1024.

**Phase 4.5 — Tiny end-to-end smoke gate.** Run Phases 2–7 wiring on ~20 utterances / 2 accents to validate the full path (features → probe → route → curve) before scaling. *Acceptance:* a (rough) curve is produced end-to-end on the tiny subset.

**Phase 5 — Baselines & references.** Implement all comparators as **score→curve** with a shared routing/eval harness: default-always, careful-always, oracle, random, **argmax-classifier (CommonAccent)**, confidence (P1-style). *Acceptance:* net-WER-vs-escalation curves for every baseline over the subset, plus oracle and floors.

**Phase 6 — The improvement: scalar probe.** Train the probe (+ learnable layer weights) to regress base-model per-utterance WER, **speaker-held-out**; calibrate; threshold. Small MLP/linear, MSE/Huber, fixed seed; save `models/probe.pt` and the learned layer-weight vector (report which layers dominate). *Acceptance:* scalar's curve computed on the held-out test fold on the same axes as the baselines.

**Phase 7 — Evaluation & analysis.** Overlay all curves; compute summary stats (net WER @ fixed escalation, area vs random); **per-accent and per-L1** deltas; substitution/deletion profile; scalar-vs-speaker leakage check; plots saved to the experiment folder. Explicitly surface conditional/negative findings and edge cases (overconfidence on accented speech; scalar collapsing to speaker id; cross-corpus calibration if CV/LibriSpeech used). *Acceptance:* a results figure + table showing scalar vs argmax vs confidence vs oracle, sliced.

**Phase 8 — (Optional / stretch) Flywheel & drift.** *Do this only if time remains; mark clearly as optional.* (a) **Hard-case mining:** log the highest-scalar and highest-misrouting-cost utterances as a mineable set for the next iteration. (b) **Drift detection:** treat the scalar's distribution as a monitoring signal — simulate drift (e.g. shift accent mix or add a read-speech batch) and show the scalar distribution flags it. Write up how eval outputs would feed the next development loop (threshold re-tune, data acquisition). *Acceptance:* a short `experiments/` report demonstrating the drift signal on a simulated shift.

**Phase 9 — Write-up (required final deliverable).** Produce the **2–3 page write-up** in `writeup/` covering exactly what the assignment asks: problem definition (what is predicted, success/failure, assumptions); baseline; the one meaningful improvement; eval plan + results summary (data, metrics, edge cases, what was learned from the comparison); failure modes / open questions / next-week plan; and the **AI-tool-use note** (delegated / accepted / corrected / rejected / how validated — synthesize from `docs/ai-use-log.md`). Keep it honest about limitations. *Acceptance:* write-up is complete, ≤3 pages, and references the results figures.

**Phase 10 — Finalize for handoff.** Finish README (install, quickstart, **full reproduction command**, results summary, hardware/runtime notes, licensing: EdAcc CC-BY-SA, CommonAccent MIT, CV/MDC terms if used). Ensure `make reproduce` runs the whole pipeline on the default subset from a clean checkout. Remove dead code, confirm no secrets/data/weights committed, final commit + push. *Acceptance:* a fresh clone + documented steps reproduces the headline curve.

---

## 8. Experiment Reporting Protocol

**After every experiment**, write a self-contained report to `experiments/EXP-<NN>-<slug>/report.md` so that an AI or human can review and summarize it in seconds. Each report contains: **date/commit hash**, **question asked**, **config** (models, subset, seed, threshold sweep), **method** (one paragraph), **results** (the curve figure + a small metrics table, incl. net WER @ fixed escalation and per-slice deltas), **what it means**, **caveats / failure modes observed**, and **next action**. Save the figures and a `metrics.json` alongside. Keep an `experiments/INDEX.md` table summarizing all runs. Reports must be skimmable and must state negative results plainly.

---

## 9. Cross-cutting Requirements

- **Reproducibility:** everything config- and seed-driven; `make reproduce` runs end-to-end on the default subset; cache expensive steps (ASR, features) so iteration is cheap.
- **Testing:** the smoke tests in §7 (WavLM finite, tiny e2e, WER sanity) live in `tests/` and run in CI-lite via `make smoke`.
- **Determinism:** fix seeds; log versions of key libs into each experiment report.
- **Honesty & defensibility:** log every non-obvious decision in `docs/DECISIONS.md`; keep `docs/ai-use-log.md` current; prefer clear negative results over inflated ones.
- **Data hygiene:** `data/`, `models/`, `.env` gitignored; only code, configs, docs, small figures, and reports are committed.
- **Assignment deliverables checklist (must all be satisfied):** problem definition ✓ · baseline ✓ · one meaningful improvement ✓ · prototype/script demonstrating both ✓ · eval plan + results ✓ · failure modes / open questions / next steps ✓ · AI-use note ✓ · ~2–3 page write-up ✓ · run instructions in README ✓.

---

## 10. Open knobs to confirm in your plan (don't block — propose defaults)

1. **Default vs careful ASR pair** (default `whisper-small` → `whisper-large-v3`; alt `turbo`).
2. **Scalar target** (default: regress base-model per-utterance WER; ablation: regress escalation-gain = default-WER − careful-WER, which is the decision-theoretically ideal target).
3. **Subset size** (propose a starting accent set + per-accent cap that runs comfortably in one sitting on 24 GB).
4. **Whether to include the Common Voice read-contrast in v1** or defer it to the optional stretch (default: defer; EdAcc core first).

Produce the plan now (task breakdown + repo tree + the four defaults above). Choose sensible values for the four defaults yourself and state them. Wait for my approval of the plan before implementing; after that, proceed autonomously per §0.1.
