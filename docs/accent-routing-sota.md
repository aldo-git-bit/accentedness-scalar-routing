# Accent Routing — State of the Art

*Knowledge base for the Deepgram accent-routing exercise. Scope: what the field has built for handling accent in speech systems, organized so that later design and evaluation decisions can point back to a specific place in the landscape. Implementation is deliberately out of scope here.*

Last updated: 2026-07-03. All external claims are linked in [References](#references).

---

## TL;DR

Accent handling is usually discussed along a single axis — **accent-agnostic vs accent-aware** — but that axis alone hides the decision that actually matters for a *product*: **whether you branch computation at inference, and on what signal.** Separating those two questions gives a 2×2 that organizes the whole field and locates the two approaches worth considering for a routing product:

1. **Accent routing** — detect the accent (classifier/adapter) and send the utterance to a specialist path. *Accent-aware, branches at inference.*
2. **Quality-driven escalation** — don't identify the accent at all; escalate to a more careful (costlier) path when an internal quality proxy (confidence, predicted WER) says we are likely failing. *Accent-agnostic, branches at inference.*

The historical literature is concentrated in the **left column** (build one model, don't branch): invariance training and accent-conditioning. The **right column** (branch at inference) is comparatively recent and thin — and the **quality-driven-escalation cell is nearly empty in the accent literature specifically**, borrowed almost entirely from adjacent work on confidence estimation and model cascades. That emptiness is an opportunity, not a gap to apologize for.

The single most important design constraint that falls out of this map: **the routing signal must be read from acoustic / self-supervised representations, not from decoded text.** A cascade that classifies accent (or estimates difficulty) from the ASR transcript has already discarded the acoustic and prosodic evidence that accent lives in. (This is the hinge to the companion note on *what accent is*.)

---

## 1. Framing: two orthogonal axes

### Axis A — Routing signal: *what decides the path?*

- **Accent identity** — an explicit accent label/embedding drives the decision (accent-aware).
- **Quality / difficulty proxy** — confidence, predicted WER, entropy, or acoustic quality drives the decision; accent is never named (accent-agnostic in the literal sense).
- **None** — no per-utterance branching; one model handles everything.

### Axis B — Response mechanism: *what changes once decided?*

- **Invariance** — one model made robust to accent, no branching (adversarial training, augmentation, meta-learning, scale).
- **Conditioning** — accent information fed into one model (embeddings, one-hot, codebooks, gating).
- **Modular adaptation** — accent-specific parameters selectively activated (adapters, LoRA experts, mixture-of-experts).
- **Model / pipeline routing** — the utterance is sent to a *different* model or an extra pass (specialist model, bigger model, more careful decoding). This is where escalation lives.

### The terminology trap (this is the thing that was unclear)

The canonical dichotomy in the ASR literature is **accent-agnostic** ("the modeling of accents inside the ASR system is not made specific") vs **accent-aware** ("additional information about the accents of the input speech is used") ([Ngo et al. 2024](#ref-tts-aug); [Prabhu et al. 2024](#ref-codebooks)). That dichotomy is about **representation and training**, *not* about inference-time branching.

"Escalate when quality degrades" is therefore **not** what the literature means by accent-agnostic. It *is* agnostic to accent (it never names one), but it is a **routing/cascade mechanism**, whereas textbook "accent-agnostic" denotes a single invariant model. The two ideas share the word "agnostic" and nothing else. The clean way to see it: *agnostic vs aware* is Axis A; *single model vs routing* is (the branching end of) Axis B. The two candidate product approaches are the two right-hand cells below — same column (they both branch), different row (accent label vs quality proxy).

### The 2×2

| | **Single model** (no inference-time branch) | **Routing / cascade** (branch at inference) |
|---|---|---|
| **Accent-agnostic** *(accent not modeled)* | **Invariance** — adversarial training, accent relabeling, contrastive/similarity losses, MAML, data augmentation, just-scale-the-data. *Indicative census: ~7 representative method families.* | **Quality-driven escalation** *(candidate approach #2)* — confidence / reference-free-WER trigger → bigger or more careful path. *Census: ~1–2, and borrowed from confidence-estimation & model-cascade literatures rather than native to accent work.* |
| **Accent-aware** *(accent modeled)* | **Conditioning** — accent embeddings, accent-specific codebooks, gating, multi-domain training, layer-wise adaptation baked into one model. *Census: ~8 representative method families.* | **Accent routing** *(candidate approach #1)* — accent classifier/adapter → specialist model; MoE / LoRA-experts / dynamic dialect routing. *Census: ~5, skewed to 2024–2026.* |

**On the numbers.** The cell counts are an *indicative census of the representative methods cataloged in §2 of this document* — they illustrate direction, not a precise bibliometric proportion. The direction is corroborated by the two field surveys: the 2021 survey observes that approaches "mostly focus on single model generalization and accent feature engineering" ([Hinsvark et al. 2021](#ref-survey2021)), i.e. the left column; and the 2025 systematic review of 58 studies / 24 datasets frames the field's core approaches as classification architectures (CNN/LSTM/Transformer/multi-embedding) feeding *accent-aware* models ([Salifu et al. 2025](#ref-review2025)), again a representation-and-conditioning story, not a routing story. Two further qualitative signals reinforce the imbalance:

- **Temporal skew.** Almost everything in the right column is 2024–2026 (MoE for accented ASR, LoRA experts, dynamic dialect routing). The left column spans a decade-plus.
- **The empty cell.** Quality-driven *accent-agnostic* routing has essentially no accent-native literature. The mechanism exists and is deployed — low-confidence utterances get passed to a different (e.g. server-side) model ([Google, context-aware confidence patent](#ref-confpatent)) — but it is framed as generic confidence routing, not as accent handling. Applying it *as* an accent strategy is close to open ground.

### The unbounded taxonomy problem

The accent-*aware* cells tacitly assume a **fixed, enumerable label set**. Real deployments break that assumption: a call centre may field speakers from 1,000+ first languages, and accent is a *continuum* (L1 × exposure × register), not a set of discrete classes. Two failures follow. First, discrete N-way classification degrades as N grows — strong at 4-way but falling toward ~51% at 7-way and ~32% at 23-way ([Ge et al. 2016](#ref-ge)) — and most accent-aware corpora are small closed sets (MCV-Accent is *five* native varieties; CommonAccent 16). Second, a specialist-per-accent is operationally infeasible, and any fixed classifier is brittle to accents unseen at training ([REDAT](#ref-redat)).

The field's response is to **stop using discrete labels**:

- **Continuous accent embeddings + basis interpolation** — represent any accent, unseen ones included, as a linear combination of a small basis; cost is O(bases), not O(accents) ([Gong/Qian 2022](#ref-layerwise)).
- **Similarity / nearest-neighbour selection** — for a new accent with no data, train on its embedding-space neighbours; embedding similarity beats geographic proximity ([AccentFold](#ref-accentfold); 120 accents, 41 held out unseen).
- **Accentedness as a scalar** — model *degree* of accent, not identity: needs no taxonomy, and doubles as a difficulty signal ([Ghorbani & Hansen 2023](#ref-gh2023)).

At scale, "accent-aware" quietly stops being *discretely* aware — soft embedding routing and accentedness scalars blur toward the agnostic/escalation column. (The agnostic single-model and quality-driven-escalation approaches never had this problem: they never enumerate accents.)

### The architectural substrate: cascade vs audio-native

The 2×2 describes routing *logic*; this axis describes the *architecture that logic runs inside*, which decides where the accent signal still physically exists to route on. Two senses of "cascade" — keep them apart:

- **Classical:** HMM–GMM (acoustic model + lexicon + LM) vs end-to-end neural ASR. Most modern accent work already sits on the E2E side (Whisper, wav2vec2, conformer-RNNT).
- **Contemporary:** a **cascaded voice pipeline** (STT → LLM → TTS, glued by *text*) vs an **audio-native / speech-to-speech (S2S)** model that maps audio→audio in one model with no text intermediary ([Coval 2026](#ref-coval); [Speko 2026](#ref-speko)).

The mechanism that matters here: cascaded pipelines discard paralinguistic cues — accent, prosody, emotion — at each inter-module *text* boundary ([FlashLabs Chroma 2026](#ref-chroma)). So in a cascade, accent routing can only live **at or before the ASR encoder**, or on a side-channel tapping the audio directly; anything downstream (LLM, TTS) sees only text and is blind to accent. Audio-native models have no text boundary, so the accent signal flows through — the "route on acoustics not text" constraint (§3) *dissolves*, and routing goes **internal** (MoE, latent conditioning, adaptive compute) rather than being an external pipeline branch.

The twist: the text bottleneck that *destroys* accent information is the same one that makes cascades enterprise-friendly — the text intermediary is what enables logging, auditing, and content-moderation *before* the user hears output, which is why cascaded still dominates enterprise voice in 2026 despite S2S's latency and prosody wins ([Coval 2026](#ref-coval); [Speko 2026](#ref-speko)). Audio-native is therefore not simply "better for accent"; it trades accent preservation against control, debuggability, and compliance.

**Consequence for this whole document: the 2×2, every method in §2, and the entire shelf in §5 are cascade-era artifacts.** Accent handling *inside* audio-native models is barely a field yet (see [§6](#6-open-problems--whats-underexplored)).

---

## 2. The four cells (method tour)

### 2.1 Left / Agnostic — Invariance (build one robust model)

The dominant historical approach: make a single model insensitive to accent.

- **Domain-adversarial training.** A gradient-reversed accent classifier forces the encoder to drop accent-discriminative information; shown to improve over standard ASR ([Sun et al. 2018](#ref-adversarial); [Na & Park 2021](#ref-nap2021)).
- **Accent relabeling / clustering.** Re-derive accent groupings from data rather than trusting self-reported labels ([Hu et al. 2020](#ref-hu2020)).
- **Similarity / contrastive losses.** Cosine or contrastive objectives pull representations of the same content across accents together, yielding "accent-neutral" models ([Ngo et al. 2024](#ref-tts-aug), and refs therein).
- **Meta-learning (MAML).** Train for *fast adaptation* to unseen accents; outperforms joint training in zero-/few-/all-shot cross-accent settings ([Winata et al. 2020](#ref-maml)).
- **Data augmentation.** Synthesize accented data to broaden coverage: unsupervised TTS augmentation ([Ngo et al. 2024](#ref-tts-aug)), synthetic cross-accent augmentation ([Klumpp et al. 2023](#ref-klumpp)), augmentation for bias mitigation ([Zhang et al. 2023](#ref-zhang-aug)).
- **Just scale the data.** The implicit Whisper thesis. Necessary but not sufficient: the 680k-hour model still degrades sharply on accented conversational speech (see [§5.1, EdAcc](#ref-edacc)).
- **Saliency-driven spectrogram masking.** A recent lightweight, model-agnostic robustness trick that masks accent-salient spectrogram regions during fine-tuning ([2025](#ref-saliency)).

*Strengths:* one model to ship; no labels or taxonomy at inference; graceful on unseen accents (especially MAML/augmentation). *Limits:* invariance discards information that could help; there is a fairness/accuracy ceiling from forcing one model to cover a heavy-tailed accent distribution.

### 2.2 Left / Aware — Conditioning (one model, told the accent)

Keep a single model but *give* it accent information rather than removing it.

- **Accent / native-language embeddings** concatenated or fused into the acoustic model ([Ghorbani & Hansen 2018](#ref-gh2018); multi-embedding + ASR, [Ghorbani & Hansen 2023](#ref-gh2023)).
- **Accent-specific codebooks** with cross-attention: learnable per-accent codebooks the decoder attends over — a strong recent accent-aware result on MCV-Accent ([Prabhu et al. 2024](#ref-codebooks)).
- **Gating mechanisms** for multi-accent adaptation that mix accent-specific and shared pathways ([Zhu et al. 2019](#ref-gate)).
- **Layer-wise fast adaptation** — adapt only the most accent-sensitive layers of an E2E model; because accents live in a *continuous embedding space*, unseen accents are handled at inference by interpolating a small set of adapter bases — a direct answer to the [unbounded-taxonomy problem](#the-unbounded-taxonomy-problem) ([Gong/Qian et al. 2022](#ref-layerwise)).
- **Accent-robust SSL embeddings** — supervised and unsupervised wav2vec embeddings that carry accent information into the recognizer ([Li et al. 2021](#ref-li2021)).
- **Multi-domain training** — train on labeled accent domains jointly so the model conditions implicitly.

*Note:* conditioning still ships as one model — the accent signal is an *input*, not a *router*. This is why it sits in the left column despite being accent-aware.

### 2.3 Right / Aware — Accent routing (detect → specialist)

The "classical" accented-ASR pipeline: an accent-identification stage selects an accent-specific system ([Hinsvark et al. 2021](#ref-survey2021)). Modern instantiations replace hard model-switching with learned modular routing:

- **Mixture-of-experts with learned routing** over accent/condition-specialized subnetworks — increases capacity without proportional compute; recent work adds intermediate-CTC supervision for accented ASR ([2026](#ref-moe)).
- **Mixture of LoRA experts** for low-resource multi-accent ASR — parameter-efficient per-accent experts instead of full specialist models ([2025](#ref-lora-moe)).
- **Dynamic dialect routing** using feature–embedding combinations, and **beam-search expert/codebook selection** when accent labels are unavailable at inference ([Prabhu et al. 2023/24](#ref-codebooks); Jie et al. 2024, via [MoE 2026](#ref-moe)).

*Strengths:* can select the *right* specialist and can act accent-specifically (e.g. an Indian-English-tuned model). *Limits:* needs an accent taxonomy, labels, and a specialist per accent; misroutes on classifier error or unseen accents; the classifier's *precision* (not accuracy) governs whether routing pays for itself. *Scaling past a fixed label set* (the [unbounded-taxonomy problem](#the-unbounded-taxonomy-problem)): swap hard argmax routing for soft routing over a continuous accent space, nearest-neighbour specialist selection, or AccentFold-style selection of which accents to train the specialist on ([Gong/Qian 2022](#ref-layerwise); [AccentFold](#ref-accentfold)).

### 2.4 Right / Agnostic — Quality-driven escalation (the near-empty cell)

Route on a difficulty/quality proxy; treat accent as one of several causes of degradation (alongside noise, far-field, codeswitching, disfluency).

- **Confidence-gated model routing.** Low utterance confidence → pass audio to a different/larger (e.g. server-side) recognizer. Deployed pattern, framed generically ([Google confidence patent](#ref-confpatent); confidence uses in [ELECTRA rescoring](#ref-electra)).
- **Reference-free WER / quality estimation as the trigger.** e-WER and successors predict error rate with no reference transcript; NoRefER provides a multilingual reference-free quality signal and can even beat raw confidence scores at flagging errors ([e-WER / system-independent WER est.](#ref-ewer); [NoRefER](#ref-norefer)).
- **Two-pass / cascaded-encoder deliberation** — a cheap streaming first pass plus a careful second pass; usually *always-on* rather than conditionally triggered, but the same machinery a conditional escalator would reuse ([two-pass RNN-T / deliberation](#ref-twopass)).

*Strengths:* no accent taxonomy or labels; generalizes past accent to any degradation; degrades gracefully to "spend more compute." *Limits:* blind to *why* it is failing, so it can only "try harder," never "try differently" — useless for accent-specific generation (TTS) and for picking an accent-tuned model. Capped by the proxy's reliability, and E2E softmax confidence is famously **overconfident** ([ELECTRA rescoring](#ref-electra); [rare-word confidence](#ref-rareword)), which is exactly why the reference-free-WER line exists.

---

## 3. Trigger mechanisms (machinery for the routing column)

The right column is only as good as its trigger. Two families:

### 3.1 Accent identity (aware trigger)

- **Pretrained accent classifiers.** CommonAccent (SpeechBrain) — ECAPA-TDNN and wav2vec2/XLSR recipes on Common Voice, up to ~95% English accent-classification accuracy; embeddings cluster by phonological similarity ([Zuluaga-Gomez et al. 2023](#ref-commonaccent)). Ready-to-use checkpoints in [§5.2](#52-pretrained-models).
- **Accent adapters.** Lightweight per-accent modules inserted in a frozen backbone; the routing decision becomes "which adapter," not "which model."
- **Beyond fixed classes.** The discrete argmax over a fixed accent set is exactly the part that fails at scale ([unbounded-taxonomy problem](#the-unbounded-taxonomy-problem)); the scalable trigger is *embedding similarity* to reference accents or a continuous *accentedness* regression, neither of which enumerates accents ([AccentFold](#ref-accentfold); [Ghorbani & Hansen 2023](#ref-gh2023)).

### 3.2 Quality / difficulty (agnostic trigger)

- **Utterance confidence** from the recognizer — cheap but overconfident for E2E models; confidence-estimation modules (CEMs) partly correct this ([ELECTRA rescoring](#ref-electra); [rare-word confidence](#ref-rareword)).
- **Reference-free WER estimators** — e-WER, e-WER3, Fe-WER, NoRefER — a more reliable, model-agnostic trigger; some operate on black-box (commercial) outputs ([e-WER family](#ref-ewer); [NoRefER](#ref-norefer)). *Aside: NoRefER was evaluated on hypotheses from commercial engines including Deepgram.*

### 3.3 Hybrid: high-precision routing with a quality backstop

The two triggers compose, and the composition directly answers the "routing errors are costly" objection. Run a *high-precision* accent classifier (§3.1) that fires **only** on the common accents it handles reliably — a selective classifier with a reject/abstain option — route those to specialists, and let the quality trigger (§3.2) catch everything it declines: rare or unseen accents, low-confidence cases, degraded audio. You buy accent-specific gains where they are cheap and safe, and a performance *floor* everywhere else, without enumerating the long tail. It is the accent instance of a standard meta-pattern — selective prediction / learning-to-defer with a confidence-gated fallback (high-confidence → fast specialized path, low-confidence → careful path), as used in hybrid dialogue routing ([Hybrid routing 2025](#ref-hybridroute)).

No single published accent-ASR system packages exactly this two-tier structure (nothing found this session), but the evidence points straight at it:

- The accent-MoE work finds hard accent classification is "neither achievable nor necessary" and that auxiliary classifiers add little over inference-time selection ([MoE-CTC 2026](#ref-moe); [Prabhu et al. 2023/24](#ref-codebooks)) — i.e. don't lean on the classifier alone; back it with a robust default.
- **Deepgram's own guidance is a gated version of this idea, ordered backstop-first**: treat accent-*robust* ASR (Nova-3) as the default that already handles most accents, and add accent *classification* only where it clears precision and ROI bars (per-class accuracy >75%, projected value >~3× cost) — because production telephony drops classifier accuracy from ~85–95% in the lab to ~55–79% under noise and codec degradation. They monitor the confidence-score *distribution* as the live signal — the same quality trigger, repurposed for drift detection ([Deepgram 2026](#ref-dg-accent)). The reframing is sharp: once the backstop is strong, the classifier's job shifts from *rescuing WER* to *enabling accent-specific action* (routing, personalization), and the quality trigger is what guarantees the floor.

*Net:* the hybrid is well-motivated and on-brand. The real design question is the **handoff policy** — an abstention threshold on the classifier, a quality threshold on the backstop, both tuned against **net WER including misrouting cost** (the evaluation gap in [§6](#6-open-problems--whats-underexplored)).

### The load-bearing design constraint

Whatever the trigger, **read it from the acoustic or SSL representation (waveform → wav2vec2/HuBERT/WavLM, or the ASR encoder states), never from the decoded hypothesis text.** Accent — and much of what makes an utterance hard — is carried in segmental and prosodic acoustics that the transcript has already thrown away. A naive cascade that classifies accent or estimates difficulty *from the transcript* is the tempting-but-wrong design this whole map rules out. A useful corollary is a **shared front end**: compute SSL features once, fan out to accent-ID/quality-estimation and to ASR — cheap, and very much in the spirit of a data factory. (Justification for this constraint is the subject of the companion note, *what accent is → architecture*.)

This constraint *is* the cascade text bottleneck (see [§1, architectural substrate](#the-architectural-substrate-cascade-vs-audio-native)): it applies precisely because a pipeline loses paralinguistics at each text boundary. In an audio-native / S2S model the constraint dissolves — there is no text boundary, so the signal is available throughout and "routing" becomes internal (MoE, latent conditioning) rather than a pre-ASR branch.

---

## 4. Cross-cut by task

The 2×2 instantiates differently per task; this is the second organizing dimension.

### 4.1 ASR
All four cells populated; the natural home of everything above. Metric currency is WER/CER, but the useful slices are the substitution/deletion profile and rare-word / named-entity error rate, where accent damage concentrates.

### 4.2 TTS — the goal inverts
For synthesis, accent is a **control target you want to render**, not a defect to normalize. So TTS lives almost entirely in the accent-*aware* cells — accent embeddings, control tokens, reference/style encoders, multi-level VAEs for controllable accent ([Melechovsky et al. 2023](#ref-vae-tts); accent conversion with discrete units, [Nguyen et al. 2024](#ref-ac-tts)). Invariance is meaningless here, and quality-driven escalation is a weak fit (at most, route to a better vocoder). Being fully accent-aware, TTS meets the same [unbounded-taxonomy problem](#the-unbounded-taxonomy-problem) and answers it the same way: continuous, speaker-agnostic accent embeddings enable *zero-shot* generation of unseen accents without one-hot labels ([AccentBox/GenAID 2024](#ref-genaid)). Worth stating explicitly so the ASR intuitions aren't over-generalized.

### 4.3 Voice agents / NLU
Inherits ASR, then compounds: upstream WER errors cascade into intent-classification and slot errors. The quality proxy can be *intent* confidence rather than transcription confidence, and escalation-to-a-bigger-model or escalation-to-human is very natural. Turn-level failure signals (repetitions, re-prompts) are additional triggers unavailable in single-utterance ASR.

---

## 5. On the shelf (datasets, models, libraries)

*Specifics are given only where verified this session; items flagged "verify" need a specifics check in the deep-dive.*

### 5.1 Datasets

**Evaluation-grade (accent-labeled, real speech):**

- **EdAcc — Edinburgh International Accents of English.** ~40 h conversational English dyads, 40+ self-reported accents across 51 first languages, per-speaker linguistic profiles, CC-BY-SA. Purpose-built as a bias benchmark: the best 680k-hour model reaches ~19.7% avg WER vs 2.7% on US clean read speech, with consistent degradation on Indian, Jamaican, and Nigerian English. Conversational domain makes it the most product-realistic eval. ([paper](#ref-edacc); [HF](#ref-edacc-hf)) **This is the primary eval anchor.**
- **Common Voice.** Massively multilingual, crowd-sourced, includes accent + locale metadata. Domain is *read* speech — important mismatch vs conversational deployment, and vs EdAcc. Contributor self-descriptions of accent are heterogeneous. ([Ardila et al. 2020](#ref-cv))
- **MCV-Accent.** Accent-annotated subsets of Common Voice used by the codebooks line ([Prabhu et al. 2024](#ref-codebooks)).
- **TIMIT** (dialectal US English, read) and **Speech Accent Archive** (single elicitation paragraph, very broad L1 coverage) — classic but narrow-domain ([Weinberger 2015](#ref-saa)).
- **AfriSpeech-200** — ~200 h, ~120 African-English accents, 2,000+ speakers, with 41 accents held out test-only. The go-to for *zero-shot / unseen-accent* evaluation — i.e. the [unbounded-taxonomy](#the-unbounded-taxonomy-problem) stress test, and the accent-aware scalability benchmark ([AccentFold](#ref-accentfold)).

**Also relevant (verify hours/license/domain in deep-dive):** AccentDB (structured non-native English) ([Ahamad et al. 2020](#ref-accentdb)); VCTK (multi-speaker English, TTS-oriented) ([Veaux et al.](#ref-vctk)); L2-ARCTIC and CORAAL (African American English) — both widely used, not verified this session.

The 2025 systematic review catalogs **24 public accent datasets** with hours/sampling-rate/demographics — the right source to mine when we build the dataset table ([Salifu et al. 2025](#ref-review2025)).

### 5.2 Pretrained models

**Accent identification:**
- `Jzuluaga/accent-id-commonaccent_xlsr-en-english` — 16 English accents, XLSR, MIT ([HF](#ref-commonaccent-hf)).
- `Jzuluaga/accent-id-commonaccent_ecapa` — ECAPA-TDNN variant.
- Smaller CV-trained classifiers (`dima806/...`, `MilesPurvis/...`, 5–6 accent classes) — lighter, coarser.

**ASR (general + specialist):**
- **Whisper** family (base/small/medium/large-v3, + turbo) — general baseline; larger sizes double as a "specialist = more compute" proxy for escalation experiments.
- **Indian-accent / Indic Whisper fine-tunes** exist but are uneven — some are genuinely Indian-accented English, others are Hindi/Hinglish; quality varies (e.g. `Tejveer12/Indian-Accent-English-Whisper-Finetuned`; Oriserve Hindi2Hinglish). Treat as a stretch comparison, not a dependable specialist ([HF search](#ref-whisper-indian)).
- **SSL encoders** for features/triggers: wav2vec2, HuBERT, WavLM, XLSR.

**Speaker-trait benchmarking:** Vox-Profile — a 2025 speech-foundation-model benchmark characterizing static traits (age, sex, **accent**) and dynamic traits (emotion, flow); relevant for a principled accent-signal probe (verify details) ([via EdAcc-related search](#ref-voxprofile)).

### 5.3 Libraries / toolkits

- **SpeechBrain** — accent-ID recipes (CommonAccent), ASR, embeddings.
- **Hugging Face Transformers + Datasets** — Whisper/wav2vec2 inference, EdAcc/Common Voice loading in a couple of lines.
- **faster-whisper (CTranslate2)** — CPU-efficient Whisper inference; the pragmatic choice if compute is modest.
- **NeMo**, **ESPnet** — full ASR/TTS toolkits if a heavier pipeline is warranted.
- **jiwer** — WER/CER computation.

*Everything above is **cascade-era ASR componentry** (see [§1, architectural substrate](#the-architectural-substrate-cascade-vs-audio-native)) — it assumes a text-emitting ASR stage to route around.*

**Audio-native / S2S models (a different shelf, largely without accent tooling yet):** Moshi (Kyutai, full-duplex), GLM-4-Voice, Qwen2.5-Omni, Baichuan-Audio, Sesame CSM, plus API-only GPT-4o-audio / Gemini Live. These preserve accent end-to-end but expose no accent-routing hooks; adapting the §2 methods to them is open work ([Coval 2026](#ref-coval)).

---

## 6. Open problems / what's underexplored

1. **The empty cell.** Quality-driven, accent-agnostic *routing* is barely explored as an accent strategy despite mature confidence/QE machinery. Framing accent as one driver of a general "difficulty→escalate" policy is close to open ground.
2. **Routing cost accounting.** The literature reports WER of the *specialist*, rarely the *net* WER of a routed system including misrouting cost and added latency/compute. Net-of-misrouting is the honest metric and it is usually missing.
3. **Unseen accents / unbounded taxonomy.** Accent is not a clean discrete variable; real speech is a continuum with L1 × exposure × register interactions. Classifier-based routing is brittle off its label set; MAML/augmentation degrade more gracefully. The accent-aware fixes — continuous embeddings, nearest-neighbour data selection, accentedness scalars (see [§1](#the-unbounded-taxonomy-problem)) — mitigate but don't close it: they shift the burden onto basis coverage and the reliability of the similarity metric.
4. **Proxy reliability.** E2E confidence is overconfident; reference-free WER estimators are better but imperfect. The trigger's calibration, not just its accuracy, determines whether escalation helps.
5. **Benchmark fragmentation.** Both surveys flag the lack of a standard accented-ASR benchmark and inconsistent metrics, which makes cross-paper comparison unreliable ([Hinsvark 2021](#ref-survey2021); [Salifu 2025](#ref-review2025)).
6. **Label quality.** Self-reported accent labels (Common Voice) are noisy and heterogeneous; relabeling/clustering exists precisely because of this.
7. **Accent in audio-native models.** Every method here targets a text-emitting cascade (see [§1, architectural substrate](#the-architectural-substrate-cascade-vs-audio-native)). Audio-native / S2S models preserve accent end-to-end but have almost no accent-handling literature — the first multi-dialectal end-to-end speech LLM (Tibetan) bills itself as a "first" as of 2026 ([Ti-Audio 2026](#ref-tiaudio)). Whether routing/escalation should become *internal* (MoE, latent conditioning, adaptive compute) in these models, and how to evaluate it without a text intermediary, is open — and squarely relevant to voice-native foundation models.

---

## 7. Implications for this exercise (non-implementation)

- The product-relevant territory is the **right column**, and the two candidate approaches are its two cells: **accent routing** (aware) and **quality-driven escalation** (agnostic). They are not competitors so much as two points on Axis A.
- Whichever is chosen, the evaluation must report **net WER including the cost of wrong routing decisions**, not the specialist's WER in isolation — this is the gap in the literature and the thing that demonstrates judgment.
- The trigger must be computed from **acoustic/SSL representations, not transcript text** — which is why the companion note on *what accent is* is a prerequisite, not a digression.
- **EdAcc** is the eval anchor (conversational, accent-labeled, documented degradation); **Common Voice** is the read-speech contrast that also exposes the train/deploy domain shift baked into off-the-shelf classifiers.
- At deployment scale, prefer **soft embedding routing or an accentedness scalar over discrete classify-then-route**: the [unbounded-taxonomy problem](#the-unbounded-taxonomy-problem) is the strongest argument against enumerating accents, and it pushes the aware approach toward the agnostic/escalation column — which is convergent evidence for the two right-column candidates rather than a discrete N-way router.

---

## References

<a id="ref-review2025"></a>**[Salifu et al. 2025]** A systematic review of accent classification techniques and datasets for inclusive speech recognition. *Int. J. Data Science and Analytics* (2026, online Nov 2025). https://link.springer.com/article/10.1007/s41060-025-00954-1

<a id="ref-survey2021"></a>**[Hinsvark et al. 2021]** Accented Speech Recognition: A Survey. arXiv:2104.10747. https://arxiv.org/abs/2104.10747

<a id="ref-tts-aug"></a>**[Ngo et al. 2024]** Improving Accented Speech Recognition using Data Augmentation based on Unsupervised TTS. arXiv:2407.04047. https://arxiv.org/html/2407.04047v1

<a id="ref-codebooks"></a>**[Prabhu et al. 2023/24]** Accented Speech Recognition With Accent-specific Codebooks. arXiv:2310.15970. https://arxiv.org/pdf/2310.15970

<a id="ref-maml"></a>**[Winata et al. 2020]** Learning Fast Adaptation on Cross-Accented Speech Recognition. arXiv:2003.01901. https://arxiv.org/pdf/2003.01901

<a id="ref-adversarial"></a>**[Sun et al. 2018]** Domain-adversarial training for accented ASR (via [Prabhu et al. 2024](#ref-codebooks) and [Ngo et al. 2024](#ref-tts-aug)).

<a id="ref-nap2021"></a>**[Na & Park 2021]** Accented speech recognition based on end-to-end domain adversarial training of neural networks. *Applied Sciences* (2021).

<a id="ref-hu2020"></a>**[Hu et al. 2020]** Clustering-based accent relabeling (via [Prabhu et al. 2024](#ref-codebooks)).

<a id="ref-klumpp"></a>**[Klumpp et al. 2023]** Synthetic Cross-Accent Data Augmentation for ASR. arXiv:2303.00802. https://arxiv.org/abs/2303.00802

<a id="ref-zhang-aug"></a>**[Zhang et al. 2023]** Exploring Data Augmentation in Bias Mitigation Against Non-native-accented Speech. IEEE ASRU 2023.

<a id="ref-saliency"></a>**[2025]** Accent-Invariant ASR via Saliency-Driven Spectrogram Masking. arXiv:2510.09528. https://arxiv.org/html/2510.09528v1

<a id="ref-gh2018"></a>**[Ghorbani & Hansen 2018]** Leveraging Native Language Information for Improved Accented Speech Recognition. Interspeech 2018.

<a id="ref-gh2023"></a>**[Ghorbani & Hansen 2023]** Advanced accent/dialect identification and accentedness assessment with multi-embedding models and ASR. *J. Acoust. Soc. Am.* 155(6). https://doi.org/10.1121/10.0026235

<a id="ref-gate"></a>**[Zhu et al. 2019]** Multi-accent adaptation based on gate mechanism. Interspeech 2019.

<a id="ref-layerwise"></a>**[Gong/Qian et al. 2022]** Layer-wise Fast Adaptation for End-to-End Multi-Accent Speech Recognition. arXiv:2204.09883 / IEEE TASLP 30.

<a id="ref-li2021"></a>**[Li et al. 2021]** Accent-Robust ASR Using Supervised and Unsupervised wav2vec Embeddings. arXiv:2110.03520. https://arxiv.org/abs/2110.03520

<a id="ref-ge"></a>**[Ge et al. 2016]** Accent Classification with Phonetic Vowel Representation (7-way ~51%; cites Choueiter et al. 23-way ~32%). arXiv:1604.08095. https://arxiv.org/pdf/1604.08095

<a id="ref-redat"></a>**[REDAT / Zhang et al. 2021]** REDAT: Accent-Invariant Representation for End-to-End ASR by Domain Adversarial Training with Relabeling (states the closed-set vulnerability of accent-specific systems to unseen accents). arXiv:2012.07353. https://arxiv.org/pdf/2012.07353

<a id="ref-accentfold"></a>**[Owodunni et al. 2024]** AccentFold: learned African-accent embeddings for zero-shot ASR adaptation; embedding similarity beats geography for train-set selection. Built on **AfriSpeech-200** (~200 h, ~120 accents, 41 unseen). EACL Findings 2024. Review: https://towardsdatascience.com/a-review-of-accentfold-one-of-the-most-important-papers-on-african-asr/ *(verify arXiv/anthology ID in deep-dive)*

<a id="ref-genaid"></a>**[AccentBox / GenAID 2024]** Towards High-Fidelity Zero-Shot Accent Generation — continuous, speaker-agnostic accent embeddings (GenAID) enabling zero-shot generation of unseen accents. arXiv:2409.09098. https://arxiv.org/html/2409.09098v1

<a id="ref-dg-accent"></a>**[Deepgram 2026]** McGillivray, B. *Accent Detection AI: How It Works and When You Actually Need It.* Deepgram (2026). Backstop-first stance: accent-robust ASR (Nova-3) as default, accent classification only past precision (per-class >75%) and ROI (>~3×) gates; lab-to-production accuracy drop ~85–95% → ~55–79%; confidence-distribution monitoring. https://deepgram.com/learn/accent-detection-ai-how-it-works

<a id="ref-hybridroute"></a>**[Hybrid routing 2025]** Hybrid AI for Responsive Multi-Turn Online Conversations with Dynamic Routing and Feedback Adaptation — confidence-thresholded routing (high-confidence → fast path, low-confidence → careful path); a general analogue of the selective-routing-with-fallback pattern. arXiv:2506.02097. https://arxiv.org/pdf/2506.02097

<a id="ref-coval"></a>**[Coval 2026]** Voice AI Models in 2026 / Speech-to-Speech vs Cascaded Voice AI. Coval (2026). Landscape of cascaded (STT→LLM→TTS) vs audio-native/S2S models (Moshi, GPT-4o, Sesame CSM…); why cascaded still dominates enterprise (observability, compliance). https://www.coval.ai/blog/speech-to-speech-vs-cascaded-voice-ai-which-architecture-should-you-deploy/

<a id="ref-speko"></a>**[Speko 2026]** Speech-to-Speech vs Cascaded Pipelines: Which Architecture for Voice AI? Speko (2026). Text intermediary enables logging/audit/moderation; S2S ~85% latency reduction but control/compliance costs. https://speko.ai/blog/s2s-vs-cascaded

<a id="ref-chroma"></a>**[FlashLabs Chroma 2026]** Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model — cascaded pipelines "discard paralinguistic cues once speech is reduced to text." arXiv:2601.11141. https://arxiv.org/pdf/2601.11141

<a id="ref-tiaudio"></a>**[Ti-Audio 2026]** Ti-Audio: The First Multi-Dialectal End-to-End Speech LLM for Tibetan — signals how nascent accent/dialect handling is inside audio-native models. arXiv:2604.11110. https://arxiv.org/pdf/2604.11110

<a id="ref-moe"></a>**[2026]** Mixture-of-Experts with Intermediate CTC Supervision for Accented Speech Recognition. arXiv:2602.01967. https://arxiv.org/html/2602.01967

<a id="ref-lora-moe"></a>**[2025]** Mixture of LoRA Experts for Low-Resourced Multi-Accent ASR. arXiv:2505.20006. https://arxiv.org/pdf/2505.20006

<a id="ref-confpatent"></a>**[Google]** Context-aware neural confidence estimation for rare word speech recognition (low-confidence → route to a different/server model). USPTO 12424206. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12424206

<a id="ref-electra"></a>**[ELECTRA rescoring]** ASR Rescoring and Confidence Estimation with ELECTRA. arXiv:2110.01857. https://arxiv.org/pdf/2110.01857

<a id="ref-rareword"></a>**[rare-word confidence]** Context-aware neural confidence estimation for rare word ASR (overconfidence of E2E models). See [Google patent](#ref-confpatent).

<a id="ref-ewer"></a>**[e-WER family]** ASR System-Independent WER Estimation (e-WER, e-WER3, Fe-WER). arXiv:2404.16743. https://arxiv.org/html/2404.16743v2

<a id="ref-norefer"></a>**[NoRefER]** Word-Level ASR Quality Estimation via a Reference-Free Metric. arXiv:2401.11268. https://arxiv.org/abs/2401.11268 — code: https://github.com/aixplain/NoRefER

<a id="ref-twopass"></a>**[two-pass / deliberation]** Deliberation Model Based Two-Pass E2E Speech Recognition (arXiv:2003.07962); Cascaded encoders for unifying streaming and non-streaming ASR (Narayanan et al., ICASSP 2021).

<a id="ref-edacc"></a>**[EdAcc paper]** The Edinburgh International Accents of English Corpus: Towards the Democratization of English ASR. arXiv:2303.18110. https://arxiv.org/abs/2303.18110

<a id="ref-edacc-hf"></a>**[EdAcc HF]** https://huggingface.co/datasets/edinburghcstr/edacc

<a id="ref-cv"></a>**[Ardila et al. 2020]** Common Voice: A Massively-Multilingual Speech Corpus. LREC 2020.

<a id="ref-saa"></a>**[Weinberger 2015]** Speech Accent Archive. George Mason University. http://accent.gmu.edu

<a id="ref-accentdb"></a>**[Ahamad et al. 2020]** AccentDB: A Database of Non-Native English Accents. arXiv:2005.07973.

<a id="ref-vctk"></a>**[Veaux et al.]** CSTR VCTK Corpus. University of Edinburgh. https://doi.org/10.7488/ds/2645

<a id="ref-commonaccent"></a>**[Zuluaga-Gomez et al. 2023]** CommonAccent: Exploring Large Acoustic Pretrained Models for Accent Classification. Interspeech 2023 / arXiv:2305.18283. https://arxiv.org/abs/2305.18283

<a id="ref-commonaccent-hf"></a>**[CommonAccent HF]** https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english

<a id="ref-whisper-indian"></a>**[Whisper Indian fine-tunes]** e.g. https://huggingface.co/Tejveer12/Indian-Accent-English-Whisper-Finetuned (uneven quality; verify per-model).

<a id="ref-voxprofile"></a>**[Vox-Profile 2025]** Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits (accent among static traits). *Details to verify.*

<a id="ref-vae-tts"></a>**[Melechovsky et al. 2023]** Learning Accent Representation with Multi-Level VAE Towards Controllable Speech Synthesis. IEEE SLT 2022/23.

<a id="ref-ac-tts"></a>**[Nguyen et al. 2024]** Accent Conversion Using Discrete Units with Parallel Data Synthesized from Controllable Accented TTS. arXiv:2410.03734.

---

*Companion notes: `accent-routing-scoping.md` (accent as instrument, task × metric matrix) · `accent-what-is-it.md` (why the routing signal must be acoustic/SSL, not text).*
