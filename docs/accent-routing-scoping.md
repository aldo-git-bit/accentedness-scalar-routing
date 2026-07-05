# Accent Routing — Problem Scoping

*Third doc in the knowledge base. It converts the assignment (`accent-routing-assignment.md`) plus the two research notes (`accent-routing-sota.md`, `what-is-accent.md`) into a **precise, scoped definition of success** for the coding phase. It defines the *design space and the constraints that shape it* — it deliberately does **not** commit to one instantiation. A companion note will sample this space into ~5 concrete problem/solution candidates.*

Last updated: 2026-07-04.

---

## 1. Purpose

Give the Deepgram reviewers, and ourselves, an unambiguous answer to three questions before any code is written:

1. **What is the problem?** (§3) — stated more precisely than the assignment does, using what we now know.
2. **What is a solution?** (§4) — including what a baseline is, what "improved" means, and how improvement is measured.
3. **What constrains and shapes the solution?** (§5–§6) — the learnings that winnow the mechanism space, plus hard scope, time, and infra limits.

Everything the two companion docs already argue is *referenced, not repeated*.

---

## 2. Deliverables & how they're judged

The assignment asks for a small repo containing: a short **problem definition**; a **baseline**; **one meaningful improvement**; a **prototype/notebook/script** demonstrating both; an **evaluation plan + results**; a **failure-modes / open-questions / next-steps** discussion; and a **note on AI-tool use**. Write-up ≈ 2–3 pages, plus run instructions.

We are judged on judgment, not system size — specifically on: whether the **task is well defined**; whether the **evaluation setup matches the actual problem**; whether **important failure cases** are identified; whether **implementation choices are pragmatic**; and whether **AI tools are used thoughtfully, not passively**. The interview (~75 min) spends its first third on what we built and the rest co-developing scenarios — so the artifacts should be *defensible and extensible under questioning*, not just complete.

Design implication: the eval harness is a **first-class deliverable**, because three of the five judging criteria (eval-matches-problem, failure cases, pragmatic choices) are demonstrated through it, not through the model.

---

## 3. Problem definition (precise)

**The assignment's framing — "detect whether a speaker belongs to a target accent category, and use that signal in a product" — buries the real object of study. The reframe:**

> We are not building an accent **label**. The label has little intrinsic value. We are building a **routing decision**, and its only value is the **change it produces in a downstream outcome**. Detection accuracy is an *instrument*; success is the *effect of acting on it*.

**What we decide:** for each utterance (or segment/stream), which processing path to take — e.g. default ASR vs. a more careful/adapted path — based on a signal read from the audio, not from the transcript.

**What counts as success — a quality objective measured against a cost axis.** Two distinct "costs" were conflated in the first draft; they live in different currencies and different places:

- **Primary objective — net WER (or CER):** the routed system's error rate vs. the baseline's, *after* absorbing the **quality cost of misrouting** — utterances the router sent down the wrong path (a non-target degraded by a specialist, or a target that missed its improvement), scored against an oracle router. Reporting the specialist's WER in isolation is the classic mistake (SOTA §6.2). Misrouting *quality* cost lives inside this axis.
- **Co-primary constraint — operational cost (latency + compute/\$):** paid on *every* escalation, right or wrong (running the trigger + running the heavier path). This is **not secondary** — it is constitutive of the problem: a router that "improves WER" by escalating most of its traffic is just *use the bigger model*, and has destroyed the entire rationale for routing, which is **selective spend**. So net WER is only meaningful *against a cost axis*; without one it is trivially maximized by always escalating. Measure it here as **escalation rate** (fraction sent to the heavy path — the compute/\$ proxy) and **added latency**. Note the structural wrinkle: a *serial* re-decode roughly doubles latency on escalated utterances, whereas a cheap trigger run before/parallel to ASR keeps overhead small — so routing **architecture is a cost lever**, not just a measurement. In this local prototype there is no API bill, so escalation rate + added latency (and relative model FLOPs/params) stand in for the \$ that is literal in production (cf. Deepgram's own "value must exceed ~3× infra cost" gate).
- **The honest deliverable is therefore a tradeoff, not a point:** a **quality-vs-cost operating curve** — WER against escalation rate / added latency — with the operating point chosen where the marginal WER gain justifies the marginal cost. A lone "net WER" number hides that you can always buy WER with more escalation.
- **Secondary (diagnostic):** the **error *profile*** where accent damage concentrates — substitution vs. deletion rate, and **named-entity / rare-word error rate** (what-is-accent §2, §5.5) — which refines the quality axis rather than standing beside it.
- **Rationale metric (why we care, not a scored target here):** at the voice-agent level, accent-driven ASR errors cascade into **intent errors, repetitions, and re-prompts** (SOTA §4.3). We scope to ASR (§6), so these motivate the work but are not what we measure.

**What counts as failure:** a router that improves classification accuracy but does **not** move net WER; or one that helps a common accent while silently degrading others (a per-slice regression); or one whose gain is erased by misrouting or latency cost.

**Assumptions (state explicitly in the write-up):** English; a cascaded/text-emitting ASR stack (so accent survives only in audio/encoder/metadata, not transcript — what-is-accent §3, SOTA §1 substrate); evaluation on real accented speech with references; self-reported accent labels treated as **noisy proxies**, not ground truth; and accent framed as **model calibration mismatch, not speaker deficit** (what-is-accent §4).

---

## 4. What counts as a solution

A solution has four parts. The improvement can come from *any* of them — better task definition, thresholding, slicing, data selection, or modeling (per the assignment).

**(a) Task definition — itself a design lever.** How we pose the routing decision is a choice with consequences, and *choosing well can be the improvement*:
- discrete **target-accent detection** (binary/one-vs-rest), vs.
- a continuous **accentedness / quality scalar** with a threshold, vs.
- **quality-driven escalation** that never names an accent (SOTA §2.4), vs.
- the **hybrid** — a high-precision selective classifier on common accents with a quality backstop for the rest (SOTA §3.3).
The §5 learnings push away from discrete N-way and toward scalar/quality/hybrid; the candidates note will pick among these.

**(b) Baseline — the simplest defensible position the improvement must beat.** Criteria: off-the-shelf, reproducible, and representative of "do nothing special." Two natural floors: a **single general ASR with no routing** (the "is routing worth anything?" floor), or **naïve argmax-classifier routing** (the "is our smarter router better than the obvious one?" floor). *Illustrative* off-the-shelf pieces (not a commitment): Whisper (`large-v3-turbo` / `small` / `medium`) for ASR; the SpeechBrain **CommonAccent** XLSR/ECAPA classifier for a naïve accent signal; an accentedness or reference-free-WER signal (e.g. NoRefER-style) for a quality trigger.

**(c) The meaningful improvement — measured as a delta.** Report the **change in the primary metric vs. the baseline, sliced by accent/accentedness**, plus the misrouting diagnostics, plotted **against the cost axis** (escalation rate / added latency — §3). A *valid* result includes **conditional or negative findings** ("routing helps Indian English but hurts US") — honest per-slice reporting is itself evidence of good evaluation. "Increased performance" = a net WER reduction that survives misrouting cost **at an acceptable escalation rate and latency**, ideally concentrated where the baseline was weakest; ideally shown as a small quality-vs-cost curve rather than a single point.

**(d) The system is a loop, not a one-shot — the flywheel (strong desideratum).** Structure the solution so evaluation *feeds development*:
- the **eval harness logs per-slice net WER + misrouting cases + low-confidence/hard examples**, so each run yields a mineable set for the next iteration (threshold re-tune, data selection, a better specialist);
- the **quality trigger doubles as a monitoring signal** — the same confidence / reference-free-WER distribution that routes also detects drift (Deepgram uses the confidence distribution exactly this way; what-is-accent §5.2, SOTA §3.3).
This is very on-brand for the role (a "data factory" with benchmarking and feedback on model outputs). The **closed loop (retraining) is a next-step, not a homework build** — but the harness should be *shaped* to support it from day one, which costs little.

---

## 5. Learnings → consequences (the winnowing)

Each load-bearing insight from the companion docs, and the mechanisms it puts **in** or **out** of scope. This is what shrinks the design space to something buildable.

| Learning (source) | Consequence | Scope effect |
|---|---|---|
| Detection is an instrument; the routing *behavior* is what matters (SOTA §1; §3 above) | Optimize and report a **downstream metric**, not F1 | **In:** net-WER eval. **Out:** accuracy-as-headline |
| Accent is acoustic and **dies at the text bottleneck** (what-is-accent §2–§3) | Read the trigger from **audio / SSL encoder / metadata side-channel**, never the transcript | **Out:** transcript/LLM-based accent detection |
| We're on a **cascaded** substrate; audio-native routing is internal & unbuilt (SOTA §1 substrate, §6.7) | Routing lives **at/before ASR** or on preserved metadata (timings, confidence, N-best, posteriors) | **Out (this project):** audio-native / S2S internal routing |
| **Unbounded taxonomy** + **misrouting is costly** + **precision > accuracy** (SOTA §1, §3.3) | Prefer **accentedness / quality scalars** and **embedding-similarity**; use the **high-precision-classifier-+-quality-backstop hybrid** | **Out:** discrete N-way, specialist-per-accent at scale |
| Routing = **selective spend**; latency & compute/\$ are co-primary (§3) | Keep the **trigger cheap**; prefer an architecture that doesn't **serially re-decode**; cap **escalation rate**; report a **quality-vs-cost curve** | **Out:** always-escalate, expensive triggers, unbounded escalation |
| Accent is **gradient & feature-specific** (what-is-accent §4) | Soft thresholds over hard argmax; a speaker can be native-vowel/L2-prosody | **In:** scalar/soft routing |
| Accent produces **systematic** errors (what-is-accent §5.4) | **Cause-matched** interventions; **N-best/phonetically-grounded** correction, never 1-best-text GER | **In:** targeted interventions. **Out:** blind 1-best "cleanup" |
| SSL **mid-to-upper layers** carry accent; pool over the utterance (what-is-accent §5.1) | Feature recipe: weighted-sum layers, utterance-level pooling, speaker-held-out | **In:** concrete feature extraction |
| Symbolic linguistic priors are an **auxiliary** for unseen accents (what-is-accent §5.6) | Optional zero-shot/interpretable add-on | **Next-steps**, not core |
| Accent is **not a deficit**; labels are **sensitive** (what-is-accent §4, §5.5) | Calibration-gap framing; speaker/corpus/channel-held-out; no profiling | **In:** eval & fairness discipline |

**Net in-scope mechanism space** (what the candidates note will draw from): a trigger computed from **acoustic/SSL features or ASR metadata** (accentedness scalar, confidence, reference-free WER, or a high-precision classifier); a **routing/escalation or hybrid** decision; **cause-matched interventions** (escalate to a stronger/adapted ASR, N-best-grounded correction, pronunciation-aware decoding); and **threshold/calibration tuning against net WER**. Everything else — TTS, audio-native internal routing, discrete specialist-per-accent, transcript-based detection, from-scratch training — is out.

---

## 6. Scope, operating principles & constraints

**Domain scope.** **ASR only — no TTS.** English. Evaluation on **real, accent-labeled speech**, conversational preferred (e.g. an EdAcc subset) with a read-speech contrast (e.g. Common Voice) available to expose the train/deploy domain shift. Binary/scalar routing, not N-way.

**Operating principles.**
1. Success is a **quality objective (net WER, misrouting cost folded in) measured against a cost axis** (escalation rate + latency) — never classification accuracy, and never WER without a cost budget, since selective spend is the whole point of routing.
2. The trigger comes from **acoustics / SSL / metadata**, never the transcript.
3. **Pretrained-first**; adaptation only if it clearly earns its keep.
4. **Iterable (flywheel):** shape the eval harness to feed the next iteration.
5. **Monitorable:** the routing trigger doubles as a drift signal.
6. **Pragmatic over maximal** — the judged qualities are judgment and fit, not scale.

**Time constraint.** Budget ≈ **5 hours of human time** (excluding Claude Code's own runtime). This is a hard shaper:
- **Use pretrained models.** No training from scratch, no full fine-tunes.
- If any training happens, keep it **light** — a linear probe, an **adapter/LoRA**, or threshold calibration — and only if it fits the budget.
- **Scope the data to a subset** (a few hundred utterances across a small set of accents) rather than full corpora.
- **Automate the eval** so iterations are cheap; prefer the off-the-shelf path whenever a step threatens the budget.

**Infra constraints.** Target hardware: **Apple Silicon M2/M4, ~24 GB unified memory, local-first.** On 24 GB the whole scoped pipeline runs locally:
- **ASR:** use **`mlx-whisper`** (Apple's MLX framework — the fast, memory-efficient path on Apple Silicon; `large-v3-turbo` ≈ 6 GB and runs well above real-time on M2/M4) or **`whisper.cpp`** (Metal + Neural-Engine, `brew install whisper-cpp`). **Avoid** the plain `openai-whisper --device mps` path — it is currently reported broken on Mac; MLX/whisper.cpp are the reliable local routes. (`faster-whisper`/CTranslate2 works cross-platform but is less optimal on Mac than MLX.) A non-Whisper alternative, `parakeet-mlx`, also runs on Apple Silicon.
- **SSL encoders & classifier** (wav2vec2 / HuBERT / WavLM, SpeechBrain CommonAccent): via **HF Transformers / SpeechBrain on the PyTorch MPS backend** (set `PYTORCH_ENABLE_MPS_FALLBACK=1` for any unsupported op). Base/large models are comfortable for inference and light adapter training at 24 GB.
- **Metrics:** `jiwer` for WER/CER.
- **Data:** downloadable and processed **locally** — EdAcc (~7 GB full; use a subset) and Common Voice subsets via `huggingface_hub` / `datasets`. Audio decode needs `ffmpeg`.
- **Fallback if local proves insufficient** (e.g. heavier adapter training): **Google Colab** (free T4 / Pro L4·A100), a **cloud GPU** (Lambda, RunPod, Modal), or **HF Inference** — but a properly subset-scoped task should not need them.

---

## 7. The scoped problem, in one paragraph

*Within a cascaded, English ASR setting, decide per-utterance — from an acoustic/SSL or ASR-metadata signal, never the transcript — whether to send audio down a more careful processing path, such that the routed system achieves a lower net WER (including misrouting cost) than a no-routing or naïve-argmax baseline **at an acceptable escalation rate and added latency**, with the gain concentrated where the baseline is weakest (accented/low-confidence speech) and no silent per-slice regressions. Prefer a scalar/quality or high-precision-classifier-with-quality-backstop trigger over discrete N-way classification; measure improvement as a per-slice delta with misrouting diagnostics, reported as a **quality-vs-cost curve** rather than a single point; and structure the evaluation so its outputs (hard cases, drift signals, per-slice metrics) can feed the next iteration. Build it pretrained-first, in ≈5 hours, on a 24 GB Apple-Silicon Mac.*

**Handoff:** the companion **candidates** note instantiates this space into ~5 concrete problem/solution pairs (specific target/trigger/baseline/improvement/data choices), each of which must satisfy the definition in §3, draw only from the in-scope mechanisms in §5, and fit the constraints in §6.

---

*Companions: `accent-routing-sota.md` (landscape, 2×2, triggers, hybrid, cascade-vs-audio) · `what-is-accent.md` (what accent is → feature/trigger/architecture/eval decisions).*
