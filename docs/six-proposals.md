# Accent Routing — Six Proposals

*Fourth doc in the knowledge base, and the **candidates** note that `accent-routing-scoping.md` §7 hands off to. It samples the scoped design space into **six concrete problem/solution pairs**, each satisfying the problem definition in scoping §3, drawing only from the in-scope mechanisms in scoping §5, and fitting the constraints in scoping §6. Each is written to the proposal template below; the doc closes with a **scored ranking matrix** (§8) across four equally-weighted criteria and a recommended pick.*

Last updated: 2026-07-04.

---

## 0. How to read this doc

Every proposal is one instantiation of the same scoped problem: *within a cascaded, English ASR setting, decide per-utterance — from an acoustic/SSL or ASR-metadata signal, never the transcript — whether to send audio down a more careful path, such that net WER (misrouting cost folded in) beats a no-routing or naïve baseline at an acceptable escalation rate.* They differ in **which lever moves** (task definition, trigger, intervention, data selection) and **where on the novelty/feasibility frontier they sit**.

**Scope note (why no audio-native proposal).** All six are cascade-era. This is not a retreat from acoustics: the "route on audio, not text" constraint (SOTA §3) is about tapping the **ASR encoder's SSL states** — where formants, VOT, prosody, and voice quality all still live — *not* the decoded transcript. Every proposal below reads its signal from the encoder or a metadata side-channel, so none sacrifices accent signal. Audio-native / S2S is a different *substrate* (no text intermediary anywhere), ruled out for this exercise because it breaks the net-WER-against-a-text-reference metric that three of the five judging criteria run through, exposes no routing internals yet (SOTA §5.3, §6.7), and does not fit 24 GB in ~5 h. The place it belongs is the "another week" discussion, not the build.

**The proposal template** (synthesized from the assignment's *what to do* list + scoping §3–§4):

1. **Thesis** — one line.
2. **Problem instantiation** — the specific routing decision: target, trigger source, what is predicted.
3. **Task definition** — which lever from scoping §4(a): binary detect / accentedness scalar / quality escalation / hybrid.
4. **Baseline** — the off-the-shelf floor the improvement must beat.
5. **The meaningful improvement** — stated as a delta, where it comes from, expected direction.
6. **Architecture** — cascade placement; trigger from which SSL layer / metadata; the intervention.
7. **Evaluation** — data subset, metrics (net WER vs cost curve, per-slice), edge cases, what a "win" is.
8. **Failure modes / open questions / next steps.**
9. **Infra** — concrete models, frameworks, memory footprint, local-vs-Colab.
10. **Human time** — John's hands-on time, *excluding* the coding agent's runtime.
11. **Flywheel** — how evals feed the next iteration.

**On time (soft comparison).** Scoping §6's ~5 h human-time cap is treated here as a *comparison target*, not a hard gate: proposals may exceed it in exchange for novelty/success, and the overage is reported in §10 so it can be weighed. Feasibility scores in §8 reflect the overage.

---

## Proposal 1 — Quality-driven escalation (the empty cell)

**1. Thesis.** Escalate on a reference-free quality signal that never names an accent; treat accent as one of several drivers of "this utterance is hard," and buy WER only where it pays.

**2. Problem instantiation.** For each utterance, decide *default path* (a small, cheap ASR) vs *careful path* (a large ASR re-decode), on a quality/confidence trigger read from the small model's metadata (softmax confidence, or a reference-free WER estimate). Accent is never predicted; it is absorbed into the difficulty signal.

**3. Task definition.** Quality escalation (scoping §4a, third lever) — the accent-*agnostic* right-column cell (SOTA §2.4), the one that is nearly empty in the accent literature.

**4. Baseline.** A *pair* that brackets the operating curve: **small-Whisper-always** (cheap floor, worse WER) and **large-Whisper-always** (expensive ceiling, best WER). The router must Pareto-dominate the naïve interpolation between them — i.e. recover most of the large-model WER at a fraction of its escalation cost.

**5. The meaningful improvement.** A trigger that escalates *selectively*. Delta = net WER reduction vs small-always, achieved at escalation rate ≪ 100 %, with the gain concentrated on the high-WER accent slices. The honest deliverable is the **quality-vs-cost curve** (net WER against escalation rate), not a point — this proposal is the cleanest embodiment of scoping §3's "selective spend is the whole point."

**6. Architecture.** Cascade. Small model decodes everything; the trigger runs on its output metadata (cheap softmax confidence) or via a reference-free estimator (NoRefER) on the hypothesis+audio; above threshold → re-decode with the large model. Note the latency wrinkle (scoping §3): a serial re-decode roughly doubles latency on escalated utterances, so the escalation rate *is* the cost axis.

**7. Evaluation.** EdAcc subset (conversational, accent-labeled — the primary anchor) + a Common Voice read-speech contrast to expose the read/conversational domain shift. Metric: net WER vs escalation-rate curve, sliced per accent. **Edge case that is the whole story:** E2E softmax confidence is *overconfident* (SOTA §2.4, §3.2), and it tends to be overconfident *exactly on accented speech* — so the cheap trigger may under-escalate the utterances that most need it. Demonstrating this failure and switching to a reference-free estimator to fix it is itself the "meaningful improvement." Win = a routed operating point strictly above the small/large interpolation line, gains concentrated where the baseline was weakest.

**8. Failure modes / next steps.** If the confidence trigger under-escalates accented speech, the curve collapses onto small-always; the mitigation (NoRefER) is the next step, and the comparison of the two triggers is a built-in result. Next: calibrate the estimator per-slice; add a cheap acoustic-quality feature so the trigger is *partly* pre-ASR (lower latency).

**9. Infra.** `mlx-whisper` `small` + `large-v3-turbo` (~6 GB, well above real-time on M2/M4); NoRefER (GitHub, reference-free QE); `jiwer`; EdAcc via `huggingface_hub`. Entirely local on 24 GB.

**10. Human time.** **~3–4 h** — the lowest of the six. Mostly plumbing (decode → trigger → re-decode → score) and a threshold sweep. Comfortably inside budget.

**11. Flywheel.** Strong. The confidence / QE distribution *is* the drift monitor — Deepgram uses exactly the confidence distribution this way (SOTA §3.3; what-is-accent §5.2). Logged low-confidence / escalated cases are the mineable set for the next iteration; the harness that scores the curve also emits the monitoring signal at no extra cost.

---

## Proposal 2 — Accentedness-scalar routing

**1. Thesis.** Replace the discrete N-way accent argmax with a continuous **accentedness scalar** read from SSL mid-upper layers, and route on a threshold; one scalar captures gradient accent *and* doubles as the quality trigger.

**2. Problem instantiation.** For each utterance, predict a scalar "how accented / how hard," pooled over the utterance from a learnable weighted sum of SSL encoder layers; route above threshold to the careful path.

**3. Task definition.** Accentedness scalar (scoping §4a, second lever; what-is-accent §5.2). This is the direct answer to the unbounded-taxonomy problem at the *trigger* level — no taxonomy is enumerated.

**4. Baseline.** **Naïve argmax CommonAccent classifier routing** — the "is a scalar actually better than the obvious classifier?" floor. (A single-ASR-no-routing floor can sit underneath for context.)

**5. The meaningful improvement.** The scalar handles what argmax cannot: gradient speakers ("native vowels, L2 prosody"), and unseen accents off the classifier's label set. Delta = net WER at a matched escalation rate, scalar-threshold vs argmax-route, sliced by *accentedness* rather than class. Expected direction: equal-or-better on seen accents, clearly better on the gradient/unseen tail, plus the elegance that the same scalar serves as the drift signal.

**6. Architecture.** Cascade. Feature recipe straight from what-is-accent §5.1: **learnable weighted sum across wav2vec2/WavLM layers** (accent content peaks ~layer 9; don't hard-pick the last layer), **mean-pool over the utterance**, **speaker-held-out**. The scalar itself is a light linear probe regressing per-utterance base-model WER (a Ghorbani-Hansen-style accentedness signal from ASR-error + AID). Intervention: escalate to a stronger ASR above threshold.

**7. Evaluation.** EdAcc, sliced by accentedness. Metric: net WER vs escalation curve; scalar-threshold vs argmax. Ablation: which layer(s) the weighted sum favours (a nice interpretable by-product). **Edge cases:** the scalar leaking *speaker identity* rather than accent (what-is-accent §4, §5.1) — mitigated by speaker-held-out splits and non-timbral embeddings; and cross-corpus calibration drift. Win = the scalar Pareto-dominates argmax routing, especially off-label-set.

**8. Failure modes / next steps.** The regression target is bootstrapped from base-model WER, so the scalar risks learning *base-model difficulty in general* rather than *accent* specifically — acceptable for routing (we want difficulty!) but worth stating honestly. Next: separate the accent component from generic difficulty; add embedding-similarity (→ Proposal 6) as a second head.

**9. Infra.** `wav2vec2-large` / `WavLM-large` via HF Transformers on the **PyTorch MPS backend** (`PYTORCH_ENABLE_MPS_FALLBACK=1`); SpeechBrain CommonAccent XLSR for the argmax baseline; `mlx-whisper` for ASR; `jiwer`. Light probe training (linear / weighted-sum only) is comfortable at 24 GB.

**10. Human time.** **~4–5 h** — feature extraction + weighted-sum probe training + threshold calibration. At the budget line.

**11. Flywheel.** Very strong. The scalar is *inherently* a difficulty-and-drift signal, and its regression target sharpens as more per-utterance WER is logged — the harness literally produces its own next training signal. Maximally iterable.

---

## Proposal 3 — Hybrid, language-tied (high-precision classifier + quality backstop)

**1. Thesis.** Run a high-precision selective classifier on *one* well-resourced accent (Indian English) that fires only when confident, route those to a specialist, and let a quality backstop catch everything else — accent-specific gains where cheap and safe, a performance floor everywhere.

**2. Problem instantiation.** Two-tier decision. Tier 1: a selective Indian-English classifier with a reject/abstain option. Tier 2 (for everything it declines — rare/unseen accents, low confidence, degraded audio): the Proposal-1 quality escalation.

**3. Task definition.** The hybrid (scoping §4a, fourth lever; SOTA §3.3) — the most on-brand cell, and the one the docs argue is well-motivated but unbuilt as a single packaged system.

**4. Baseline.** Two floors: **single-ASR-no-routing**, and **naïve argmax-route-to-specialist** (no abstention, no backstop) — so the improvement isolates the value of *selectivity + backstop*, not just "having a specialist."

**5. The meaningful improvement.** The **handoff policy**: an abstention threshold on the classifier × a quality threshold on the backstop, both tuned against net WER including misrouting. Delta = net WER gain on the Indian-English slice *with no silent regression on the tail*. This is the design question the literature leaves open (SOTA §3.3): "don't lean on the classifier alone; back it with a robust default."

**6. Architecture.** Cascade. Tier 1: CommonAccent XLSR, thresholded for *precision* (abstain below) — precision, not accuracy, governs whether routing pays (SOTA §2.3). Specialist: an Indic-Whisper fine-tune (treated as a *stretch* comparison — quality is uneven; some are Hindi/Hinglish, not Indian-accented English). Tier 2: the Proposal-1 escalation as the floor.

**7. Evaluation.** EdAcc (contains Indian English) with the other accents as the tail. Metric: net WER overall + Indian-English slice + explicit tail-slice check for regressions. **Edge cases:** classifier precision collapses under telephony noise (Deepgram's lab-to-production ~85–95 % → ~55–79 %, SOTA §3.3); and the specialist may *not actually beat* a large general Whisper on Indian English — which must be verified before the tier earns its place.

**8. Failure modes / next steps.** The load-bearing risk is specialist quality: if Indic-Whisper doesn't beat large-general on the slice, Tier 1 is dead weight and the proposal degrades to Proposal 1. Verify this *first*. Next: swap the hard specialist for a LoRA adapter (SOTA §2.3); add a second high-precision accent once the two-tier scaffold works.

**9. Infra.** CommonAccent XLSR (HF); an Indic-Whisper fine-tune (verify per-model — quality varies); `mlx-whisper large-v3-turbo`; NoRefER for the backstop; `jiwer`. Local on 24 GB, though the two-tier plumbing is the heaviest of the "safe" proposals.

**10. Human time.** **~5–6 h** — two tiers + policy tuning + the mandatory specialist-verification step. Slightly over budget; the overage buys the on-brand hybrid.

**11. Flywheel.** Very strong and the most *literally* a data factory: the abstained + backstopped cases are precisely the mineable set (what to add a specialist for next), and classifier precision is the monitorable production signal. This is scoping §4d's flywheel made concrete.

---

## Proposal 4 — Neuro-symbolic vowel-space probe

**1. Thesis.** A small, interpretable **vowel-space-distortion probe** (does the tense/lax high-front /iː/–/ɪ/ contrast collapse?) on frozen SSL features, used as an accentedness signal *and* a zero-shot L1-transfer prior — repurposing Mispronunciation-Detection machinery as a *routing* signal.

**2. Problem instantiation.** Predict, from a phonological probe on wav2vec2 layer ~9, whether specific L1-predictable contrasts have collapsed; route on the resulting interpretable accentedness score, and use *which* contrast collapsed to pick a cause-matched intervention.

**3. Task definition.** Accentedness scalar with symbolic structure (scoping §4a + what-is-accent §5.6). The novel move is not the probe (MDD/CAPT built it) but pointing it at *routing* rather than learner-scoring.

**4. Baseline.** The **black-box accentedness scalar (Proposal 2)** or the argmax classifier — the "does interpretable phonological structure beat the black box?" comparison.

**5. The meaningful improvement.** Two payoffs the black box can't give: (i) *interpretability → cause-matched intervention* — "merges /ɪ/–/iː/, epenthesizes s-clusters" routes to pronunciation-aware decoding, not a blind "try harder" (what-is-accent §5.4); (ii) *zero-shot reach* — a Spanish vowel chart predicts the collapse *before any Spanish-accented data is seen*, a genuinely different mechanism from embedding similarity. Delta measured on an unseen-L1 slice + on whether the probe's specific-error prediction matches the actual confusion profile.

**6. Architecture.** Cascade. Probe: read vowel structure *off the encoder* (never recompute F1/F2 by hand — formant extraction is brittle, what-is-accent §5.6), a light probe on frozen wav2vec2 layer ~9, focused narrowly on the high-front contrast (the highest-signal, most L1-predictable, most ASR-damaging, most compositional entry point).

**7. Evaluation.** EdAcc + a Spanish- or Indian-accented slice where the collapse is predictable. Metric: net WER + does the probe *correlate with the specific error type* it predicts. **Edge cases / frank cautions (what-is-accent §5.6):** on seen, well-resourced accents it likely *won't beat* a good DL classifier (the bitter lesson — the SSL model already represents the merger); the canonical-reference trap (CMUdict is itself accent-biased); don't build the full rule repository (too many rules *drops* accuracy).

**8. Failure modes / next steps.** Highest-risk of the six: the win must come from interpretability / unseen / efficiency, none of which is a single headline WER number, and demonstrating it convincingly in a few hours is hard. This is why the docs scope it as a *framing / "another week"* argument with an optional narrow demonstrator — not the core build. Next: encode a handful of L1 signatures as phonological-feature templates, match against probes for interpretable zero-shot accent ID.

**9. Infra.** `wav2vec2-large` via HF Transformers MPS; a light phonological probe (small train); `mlx-whisper`; `jiwer`; a reference vowel-space. Local, but the probe design/validation is the real cost, not the compute.

**10. Human time.** **~6–8 h — over budget**, and closest to research rather than plumbing. Reported honestly: this is the proposal whose feasibility the soft-time allowance most affects.

**11. Flywheel.** Medium. The probe's interpretable outputs feed *intervention selection* well, but building and validating the probe is more one-shot research than an iterable harness; the loop is weaker than the scalar or embedding proposals.

---

## Proposal 5 — N-best / phonetically-grounded generative error correction

**1. Thesis.** Route accented / low-confidence utterances to **generative error correction over the ASR N-best**, and demonstrate that the tempting 1-best *text* cleanup over-corrects accented speech toward the majority variety — an accuracy *and* fairness failure.

**2. Problem instantiation.** The intervention *is* post-ASR correction. Escalate hard/accented utterances to an LLM corrector; the design variable is what the corrector sees — N-best + phonetic/confidence metadata vs 1-best transcript alone.

**3. Task definition.** Quality escalation → GER intervention (scoping §4a + what-is-accent §5.4). The sharpest *case study* of the text-bottleneck thesis in the whole knowledge base.

**4. Baseline.** **1-best text GER** — an LLM cleaning the transcript. This is the tempting-but-wrong design the docs single out.

**5. The meaningful improvement.** Feeding the corrector the **N-best** (which preserves the acoustic ambiguity accent introduces — the multiple words the audio was consistent with) beats 1-best, which over-corrects (what-is-accent §5.4; HyPoradise shows GER-over-N-best surpasses the re-ranking oracle). Delta = WER on the accented slice, N-best vs 1-best, *plus a measured over-correction rate* — does 1-best regress legitimate dialect toward standard written form. The over-correction demonstration is the headline finding, and it lands whether or not the WER delta is large.

**6. Architecture.** Cascade. Whisper emits an N-best via beam search; an LLM maps N-best (+ token confidences) → corrected transcript; routing sends only low-confidence/accented utterances to GER. Never 1-best alone (what-is-accent §5.4).

**7. Evaluation.** EdAcc accented slices. Metrics: net WER + an over-correction / fidelity measure. **Edge cases:** GER hallucination; Whisper's N-best may be too narrow to carry the ambiguity; a weak local corrector. The fairness framing (correcting accent toward a default variety is the opposite of the not-a-deficit stance) is the point to carry into the interview.

**8. Failure modes / next steps.** The result could be *negative* (GER doesn't move WER, or the local LLM is too weak) — but a clean demonstration of the text-bottleneck thesis is itself a defensible result. Next: **accent-conditioned GER** (currently underexplored, analogous to code-switching GER), fine-tuned on logged corrections.

**9. Infra.** `mlx-whisper` with beam search for N-best; a local LLM for correction (Qwen/Llama via MLX or Ollama) *or* the Claude API if local memory is tight; `jiwer`; optionally HyPoradise data. The corrector is the memory pressure point on 24 GB — the API fallback keeps it local-first-ish.

**10. Human time.** **~5–6 h** — N-best extraction + GER pipeline/prompt + the over-correction metric. Slightly over budget.

**11. Flywheel.** Strong. Logged corrections + flagged over-corrections are exactly the training set for an accent-conditioned corrector; the next-step (fine-tune the GER model on mined cases) is unusually concrete.

---

## Proposal 6 — Embedding-similarity zero-shot routing

**1. Thesis.** Route by **similarity to reference-accent embeddings**, not argmax over a fixed label set, and evaluate on *held-out unseen accents* — the most direct attack on the unbounded-taxonomy problem, and the AccentFold data-selection loop made into a router.

**2. Problem instantiation.** For each utterance, find its nearest reference accents in embedding space and route to the specialist / data selection its neighbours imply — *including for accents unseen at training time*, which argmax cannot handle at all.

**3. Task definition.** Embedding-similarity trigger (scoping §4a "beyond fixed classes"; SOTA §3.1; AccentFold). The unbounded-taxonomy insight — your central framing — instantiated directly.

**4. Baseline.** **Argmax classifier routing**, which collapses off its label set — the "does similarity degrade gracefully where argmax falls off a cliff?" comparison.

**5. The meaningful improvement.** On a held-out unseen-accent split, similarity routing degrades gracefully where argmax collapses. Delta = net WER on *unseen* accents, similarity vs argmax, plus the AccentFold ablation that **embedding similarity beats geographic proximity** for choosing training neighbours. Expected direction: comparable on seen accents, decisively better on the 41 unseen.

**6. Architecture.** Cascade. Embeddings from CommonAccent XLSR (or an AccentFold-style space); route by cosine to reference-accent centroids; intervention = select the specialist / training data by embedding neighbourhood. All from the encoder side (SOTA §3 constraint respected).

**7. Evaluation.** **AfriSpeech-200** (~120 African-English accents, 41 held out test-only) — the purpose-built unbounded-taxonomy stress test. Metric: net WER on seen vs unseen, similarity vs argmax; the geography-vs-embedding ablation. **Edge cases:** the reliability of the similarity metric and basis coverage (the residual open problem, SOTA §6.3); embeddings leaking speaker identity; AfriSpeech-200's size (subset it).

**8. Failure modes / next steps.** AfriSpeech-200 download/handling is the main friction (subset via `datasets`); the unseen-accent gain could be modest if the embedding space is speaker-dominated. Next: this *is* the acquisition loop — eval on unseen accents tells you which neighbours to acquire, closing the data-factory flywheel.

**9. Infra.** CommonAccent XLSR embeddings (HF); AfriSpeech-200 subset via `huggingface_hub`/`datasets`; `mlx-whisper`; `faiss` or `sklearn` for nearest-neighbour; `jiwer`. Local with subsetting; no training required (embeddings are frozen), which offsets the data-handling cost.

**10. Human time.** **~5–6 h** — embedding extraction + NN routing + the unseen-split evaluation. Slightly over budget, mostly in data handling.

**11. Flywheel.** Very strong — arguably the strongest. This is *literally* the AccentFold data-selection loop: the evaluation on unseen accents directly answers "which data/accents to acquire next," which is the exact charter of the "data factory" role. Eval feeds acquisition by construction.

---

## 8. Ranking matrix

Four criteria, **equal weight**, each scored 1–5 (5 = best). Feasibility reflects the soft-time overage from §10. Aggregate is the simple sum (max 20).

| # | Proposal | Feasibility | Novelty | Success | Flywheel | **Agg** | Rank |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | Accentedness-scalar routing | 4 | 3 | 4 | 5 | **16** | **1 (tie)** |
| 6 | Embedding-similarity zero-shot | 3 | 4 | 4 | 5 | **16** | **1 (tie)** |
| 1 | Quality-driven escalation | 5 | 2 | 4 | 4 | **15** | 3 (tie) |
| 3 | Hybrid, language-tied | 3 | 4 | 3 | 5 | **15** | 3 (tie) |
| 5 | N-best / phonetic GER | 3 | 4 | 3 | 4 | **14** | 5 |
| 4 | Neuro-symbolic vowel probe | 2 | 5 | 2 | 3 | **12** | 6 |

**Per-criterion leaders** (since the criteria were weighted equally, here is who tops each, in case you want to reweight):

- **Feasibility:** P1 (5) — pure plumbing, all off-the-shelf, comfortably inside budget.
- **Novelty:** P4 (5) — repurposing MDD phonological probing as a routing signal is the genuinely new move; P3/P5/P6 follow at 4.
- **Success:** P1/P2/P6 (4) — well-motivated mechanisms with published support (confidence routing; accentedness correlates with perception; AccentFold's embedding>geography result).
- **Flywheel:** P2/P3/P6 (5) — the scalar that sharpens its own target; the abstain/backstop mineable set; the acquisition loop.

### Score rationale (the contestable calls)

- **P1 novelty = 2, not lower:** the *mechanism* is standard confidence routing, but applying it *as an accent strategy* is the near-empty cell (SOTA §2.4) — novel in framing, not machinery.
- **P3 success = 3:** hinges entirely on the Indic-Whisper specialist actually beating large-general on Indian English, which is unverified and uneven. If it fails, P3 degrades to P1. This is the score most likely to move after a 30-minute specialist check.
- **P4 success = 2 and feasibility = 2:** the bitter lesson caps the WER payoff on seen accents, and the real value (interpretability, zero-shot) is hard to cash into a headline number in the time. High novelty can't rescue the aggregate.
- **P6 feasibility = 3, not 4:** no training needed (a plus), but AfriSpeech-200 handling is real friction that P1/P2 don't carry.

### Recommendation

**Build P6 (embedding-similarity zero-shot), or P2 (accentedness scalar) if you want the more feasible co-leader.** They tie at 16, and the tie-break is about *what you want to demonstrate*:

- **P6** most directly instantiates your central framing insight (the unbounded taxonomy) and has the strongest, most on-brand flywheel (the data-acquisition loop *is* the "data factory" charter). It is the highest-impact interview artifact. Cost: AfriSpeech-200 handling.
- **P2** is the pragmatic co-leader — more feasible, and the scalar's dual use as *both* router and drift signal is elegant and easy to defend under questioning. It also composes cleanly into the others.

**Keep P1 as the safety floor.** If the ~5 h budget is genuinely threatened, P1 is the highest-feasibility build and is *literally* the honest core scoping §3 describes (selective spend, quality-vs-cost curve). It is also a component of P3 and a natural backstop for P2/P6 — so building it first is never wasted.

**Treat P4 as the "another week" bet.** Its novelty is real and it is a strong interview *scenario* topic ("marshal linguistic structure to help the model"), but it is a research project, not a homework build — exactly where what-is-accent §5.6 scopes it.

### These compose (worth stating in the write-up)

The six are not fully independent. P2's scalar can *be* P1's trigger; P1 is *inside* P3 as the backstop; P6's similarity head can *feed* P2's scalar. The strongest single deliverable is arguably a staged build — **P2's scalar trigger + P1's escalation + P3's selective-classifier tier** — which is exactly the SOTA §3.3 hybrid. That is more than 5 h, so the honest move for the homework is to build one cleanly (P6 or P2), and present the composition as the "with another week" architecture. This framing turns the six proposals from a menu into a *roadmap*, which reads better under interview questioning than any single point solution.

---

*Companions: `accent-routing-scoping.md` (the scoped design space these six sample) · `accent-routing-sota.md` (landscape, 2×2, triggers, hybrid, cascade-vs-audio) · `what-is-accent.md` (what accent is → feature/trigger/architecture/eval decisions).*
