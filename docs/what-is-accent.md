# What Accent Is — and What It Means for Implementation

*Companion to `accent-routing-sota.md`. That note surveys **what exists**; this one says **what accent actually is**, but only to the depth that changes an engineering decision. Every claim below is here because it moves a build choice — feature/model selection, trigger design, cascade-vs-audio, interventions, or evaluation. Section references like "SOTA §3.3" point to the companion.*

Last updated: 2026-07-03. External claims linked in [References](#references).

---

## 0. Goal

We have to make a series of decisions to get higher performance out of accent routing (or quality routing). Those decisions are only as good as our model of what accent *is* in the signal. This document builds that model just far enough to be useful, then spends most of its length ([§5](#5-implementation-decisions)) cashing it out into decisions. The linguistics is compressed into one table; the engineering is the point.

---

## 1. Working definition

**Accent is the systematic phonetic, phonological, and prosodic shaping of speech that makes a speaker sound like they come from a particular linguistic, regional, social, or language-learning background.** It is a *structured system*, not random deviation or "mispronunciation" — in L2 speech it often reflects transfer from a first language; in regional speech it is the native phonology of a variety.

The operational point for us: accent is not located in one place. It is distributed across four levels, and only the middle two are things a model can touch:

1. **The speaker's linguistic system** — phonology, phonetics, prosody, cue weighting (the cause).
2. **The acoustic signal** — formants, VOT, F0, duration, spectra, rhythm (what the model observes).
3. **The model's expectations** — what counts as "different" depends on the reference variety the model was trained on.
4. **The social interpretation** — the label "French accent" is a human reading of a structured pattern; accent indexes region, class, ethnicity, identity.

A model never hears "accent." It observes a distribution of acoustic and temporal cues and maps them against learned expectations. "French-accented" is our name for a pattern the model represents as *vowel space shifted, VOT shorter, /r/ realized differently, less function-word reduction, intonation displaced*.

---

## 2. What accent is made of (the decision-relevant decomposition)

Accent is a **probabilistic configuration of many weak cues**, not a checklist — no single feature is decisive, and a model detects it by aggregating evidence across layers. The layers, and the one thing about each that matters for us — whether it **survives the ASR text bottleneck** (see [§3](#3-what-is-detectable-from-where) / SOTA §1 substrate):

| Layer | What it is | Acoustic correlate | Survives text? | Primary decision it touches |
|---|---|---|---|---|
| **Vowels** | height/backness/tenseness/diphthongization shifts | F1/F2/F3, formant trajectories, duration | ✗ | feature source; ASR confusion slices |
| **Consonants** | place/manner/voicing, rhoticity, retroflexion, aspiration | VOT, burst, closure, F3, frication spectra | ✗ | feature source; pronunciation-aware decoding |
| **Phonemic mapping** | merged/substituted contrasts (ship–sheep, think–sink) | category overlap, shifted centroids, cue reweighting | partial (as *errors*) | accent-adapted ASR; lexicons; correction priors |
| **Allophony** | context rules (flapping, aspiration, l-darkening) applied/not | timing, spectral context variation | ✗ | encoder-level features; decoding |
| **Phonotactics** | cluster repair, epenthesis, coda deletion (*strike*→*sutoraiku*) | insertion/deletion, cluster simplification | partial (as *errors*) | N-best breadth; pronunciation variants |
| **Prosody** | stress placement, rhythm, intonation, pitch range | F0 contour, duration, energy, interval variability | ✗ | endpointing/turn-taking; TTS; trigger features |
| **Reduction / coarticulation** | how much *gonna/wanna* reduction; connected-speech processes | phone reduction/deletion, transition shape | ✗ | features; rhythm modeling |
| **Cue weighting** | which acoustic cue signals a contrast (aspiration vs voicing) | shifted cue distributions the ASR expects | ✗ | why a stronger/adapted acoustic model helps |
| **Voice quality** | breathy/creaky/nasal, spectral tilt | HNR, spectral tilt, phonation measures | ✗ | features; embeddings; not text |
| **Fluency (adjacent, *not* accent)** | pauses, hesitation, self-repair, rate | pause duration, disfluency rate | partial | routing feature — but disentangle from accent |
| **Lexicon / morphosyntax (dialect, *not* accent)** | *hoagie*, *y'all*, *might could*, double negation | — (lexical/grammatical) | ✓ | text/LLM-detectable; NLU, localization |

The pattern in the "survives text?" column is the whole game: **narrow accent lives almost entirely in acoustics that the transcript discards; only dialect/language-variety markers survive into text.** Everything in [§5](#5-implementation-decisions) follows from that column.

---

## 3. What is detectable from where

What you can route on depends on where in the stack you tap. Three regimes:

- **Raw audio / SSL encoder states** — everything in the table: segmental acoustics, VOT, formants, prosody, voice quality, coarticulation. This is where accent is fully available. Concretely, accent phenomena are recoverable from frozen encoder states as *structured gradient* variation — even African-American-English consonant-cluster reduction leaves cues to the underlying stops in wav2vec2/Whisper encoder layers ([Probing CCR 2026](#ref-ccr)).
- **ASR transcript (1-best text)** — words, some disfluencies, lexical/grammatical dialect markers, and ASR errors. It does **not** contain vowel quality, aspiration, rhoticity, prosody, voice quality, or allophony. So text-only LLMs can spot *dialect* but are poor instruments for *accent* in the narrow sense.
- **ASR metadata side-channel** — word timings, token confidence, N-best hypotheses, phone posteriorgrams, speaker/accent embeddings, prosody features. This is where most real cascaded voice systems live: not pure audio, not pure text, but **text plus a speech-metadata side-channel** that partially re-exposes the acoustics.

Measurable acoustic correlates worth knowing (the useful half of the original's table):

| Accent dimension | Signal correlate |
|---|---|
| Vowel quality | F1/F2/F3, formant trajectories |
| Aspiration | VOT, burst-to-voicing interval |
| Rhoticity | F3 lowering, r-colored vowels |
| Stress / rhythm | duration, pitch, intensity, vowel-interval variability |
| Intonation | F0 contour, boundary tones |
| Voice quality | spectral tilt, harmonics-to-noise, creak/breathiness |
| Phonotactics | epenthesis, deletion, cluster simplification |

---

## 4. Accent vs its neighbours (collapsing them causes bugs)

Four things get conflated; keeping them apart is load-bearing because they live in different regimes and demand different handling.

- **Accent** = the pronunciation/prosody system. Mostly **acoustic**; dies at the text bottleneck.
- **Dialect** = accent **+** lexicon + morphosyntax + pragmatics (*hoagie*, *might could*). The added layers **survive text** → text/LLM-detectable; relevant to NLU/localization, not to acoustic routing.
- **L2 / foreign-accented speech** = cross-linguistic transfer + acquisition effects. Same acoustic dimensions as regional accent, different *source*; often co-occurs with fluency effects.
- **Fluency** (pauses, hesitation, rate) and **channel** (mic, codec, noise, speaker identity) are **not** accent but correlate with it. This is the dangerous one: accent classifiers frequently learn **speaker identity** instead of accent ([Yang & Hansen 2023](#ref-yang)), and can latch onto channel. That confound dictates evaluation design ([§5.5](#55-evaluation--fairness)).

And the framing that matters ethically and technically: **accent is not a deficit.** A regional speaker is not "less accurate"; the ASR is *less well calibrated* to their variety because training data under-represents it. The product problem is model calibration mismatch, not speaker deficiency.

---

## 5. Implementation Decisions

The heart of the document. Each decision is stated as **accent fact → consequence → recommendation**.

### 5.1 Feature / model selection — *what to compute the signal from*

- **Fact:** accent is acoustic and distributed ([§2](#2-what-accent-is-made-of)); it is fully present only in the audio/encoder, and it is recoverable there as structured gradient variation ([Probing CCR 2026](#ref-ccr)).
- **Consequence:** derive the accent/accentedness signal from **SSL encoder hidden states** (wav2vec2 / HuBERT / WavLM / XLSR), never from transcript text, and don't bother reconstructing hand-crafted MFCCs — the SSL features already encode the phonetic and prosodic content ([Yang & Hansen 2023](#ref-yang)).
- **Recommendations (concrete):**
  - **Which layer:** don't blindly take the last layer. Phonetic/accent content peaks in **mid-to-upper** layers — accent-specific phoneme structure is strong around **wav2vec2 layer 9** (of 12), and AID fine-tuning sharpens the **top two** layers; HuBERT/WavLM retain phonetic content **through their final layers**, while prosody/suprasegmentals sit in the **middle third** ([Yang & Hansen 2023](#ref-yang); [suprasegmental probing 2024](#ref-supra); [layer-stratified analyses](#ref-orth)). Safest default: a **learnable weighted sum across layers** (SUPERB-style) rather than hard-picking one.
  - **Pooling:** accent is distributed over the utterance, so **mean-pool over time** to an utterance-level vector; word-level windows carry too little evidence.
  - **Prosody:** SSL mid-layers already carry abstract stress/rhythm/intonation (not just raw F0), so you usually don't need separate prosodic front-ends — though F0/duration/energy remain useful *interpretable* complements for a difficulty/endpointing signal.
  - **Confound guard:** because these features leak **speaker identity**, prefer speaker-invariant / non-timbral embeddings where accent-not-speaker is the target, and always evaluate speaker-held-out ([Yang & Hansen 2023](#ref-yang); non-timbral accent-ID, [Orange 2026](#ref-nontimbral)).

### 5.2 Trigger mechanism — *what decides to route*

- **Fact:** accent is **gradient**, **feature-specific**, and labelled by a **fuzzy, unbounded** taxonomy ([§4](#4-accent-vs-its-neighbours-collapsing-them-causes-bugs); SOTA §1 unbounded-taxonomy).
- **Consequence:** a discrete N-way accent argmax is the wrong trigger shape — it neither scales nor captures "native vowels but L2 prosody."
- **Recommendations:**
  - Prefer an **accentedness scalar** or **embedding-similarity** trigger over class prediction (SOTA §3.1 "beyond fixed classes"). Accentedness derived from ASR error + AID signals correlates with human perception ([Ghorbani & Hansen 2023](#ref-gh2023)) — and a scalar "how accented / how hard" *is* a difficulty proxy, i.e. it doubles as the quality trigger.
  - Use the **hybrid** of SOTA §3.3: a high-precision selective classifier on the common accents you can serve, with a **quality backstop** for everything else. This is directly motivated by "accent is distributed + labels fuzzy + gradient."
  - Whatever the trigger, compute it from **acoustics/SSL, not decoded text** (SOTA §3 constraint) — the [§2](#2-what-accent-is-made-of) table shows the transcript has already dropped the evidence.

### 5.3 Cascade vs audio-native — *where routing can physically happen*

- **Fact:** the text bottleneck discards exactly the prosody, voice-quality, and allophony rows of the [§2](#2-what-accent-is-made-of) table (SOTA §1 substrate).
- **Consequence & recommendations:**
  - In a **cascaded** pipeline, the accent/quality trigger must live **at or before the ASR encoder**, or on a **preserved metadata side-channel** ([§3](#3-what-is-detectable-from-where)): word timings, token confidence, N-best, phone posteriors, prosody. Anything downstream (LLM, TTS) sees only text and is blind to accent.
  - In an **audio-native / S2S** model there is no text boundary — the signal survives end-to-end and "routing" becomes **internal** (MoE, latent conditioning, adaptive compute). This is largely open territory (SOTA §6.7).
  - Corollary: a **shared front end** (compute SSL features once; fan out to accent/quality estimation *and* ASR) is the efficient design and fits a data-factory setting.

### 5.4 Interventions, mapped to cause — *what to do once routed (incl. error correction)*

Accent produces **systematic** errors, so match the intervention to the *kind* of accent effect rather than firing a generic "try harder":

| Accent effect | Symptom | Intervention |
|---|---|---|
| Phonemic merger / cue reweighting | plausible-but-wrong words (ship/sheep) | accent-adapted ASR; pronunciation-aware decoding; lexicon augmentation |
| Phonotactic repair (epenthesis/deletion) | cluster/coda errors | wider N-best; pronunciation variants |
| Prosodic (rhythm/intonation) | endpointing/turn-taking failures, not WER | tune VAD/endpointer; adjust TTS; usually *not* a re-decode |
| Broad degradation / low confidence | high WER, low confidence across the board | escalate to stronger/bigger ASR; pass confidence to LLM; confirm high-impact slots; human handoff |

**Error correction as an intervention (and a clean case study of the text-bottleneck thesis).** A post-ASR correction stage — today usually **generative error correction (GER)**: an LLM maps the ASR **N-best** list to a corrected transcript ([HyPoradise](#ref-hypo); [GenSEC/FlanEC](#ref-flanec)) — is a *cheap, text-level* alternative to re-decoding, and it can surpass the oracle of N-best re-ranking. Because accent yields **systematic** confusions, it is an obvious correction target, and **accent-conditioned GER** is a natural (currently underexplored) extension, analogous to code-switching GER ([Chen et al. 2023](#ref-csger)) and phonetic/rare-word GER ([2025](#ref-phoneticger)).

But here the [§2](#2-what-accent-is-made-of) thesis bites hard, and it is the sharp point worth carrying into the interview:

> **Pure 1-best *text* correction cannot fix accent errors, and tends to make them worse.** The acoustic evidence that would disambiguate ship/sheep is already gone, and text-only GER **over-corrects** — regressing accented/dialectal input toward the majority/formal written variety, *reducing* fidelity to what was actually said ([phonetic GER 2025](#ref-phoneticger)). That is an accuracy failure *and* a fairness failure (it "corrects" legitimate accent/dialect toward a default variety — the opposite of the not-a-deficit framing in [§4](#4-accent-vs-its-neighbours-collapsing-them-causes-bugs)).

The fix is the same recommendation as everywhere else: give the corrector the **N-best plus phonetic/confidence metadata** (or encoder features), never the 1-best alone. The N-best preserves the acoustic ambiguity accent introduces (multiple candidate words the audio was consistent with), which is exactly why GER over N-best beats 1-best correction ([HyPoradise](#ref-hypo)). Error correction is therefore one more instance of **preserve the acoustic side-channel** — and in an audio-native model, where there is no separate transcript, "correction" folds back into generation and this whole failure mode disappears.

*Net:* error correction is a real, cheap intervention, but only useful for *accent* when it is N-best/phonetically grounded — and it is where the temptation to "just clean up the text" does the most damage.

### 5.5 Evaluation & fairness — *how to measure honestly*

- **Gradient** → slice results by **accentedness**, not only by accent class.
- **Confounded with speaker/channel** → **speaker-held-out, corpus-held-out, channel-controlled** splits are mandatory, because accent models otherwise learn speaker identity ([Yang & Hansen 2023](#ref-yang)).
- **Damage concentrates** in phoneme confusions, rare words, and named entities → report the **substitution/deletion profile and entity-error rate**, not just overall WER (SOTA §4.1).
- **Routing has a cost** → report **net WER including misrouting**, not the specialist's WER in isolation (SOTA §6.2).
- **Not a deficit** → frame the target as **closing the calibration gap on under-served varieties**; and because accent labels proxy for ethnicity/nationality/class, constrain any product use to **performance improvement, not profiling**.

### 5.6 Symbolic linguistic priors as an auxiliary signal (esp. for the unbounded taxonomy)

Everything in §5.1–§5.2 learns the accent signal *from data* (SSL features, embeddings, accentedness). A complementary — and strictly **optional** — approach marshals explicit linguistic knowledge (vowel charts, allophonic rules, L1→L2 transfer patterns) as a **symbolic prior working alongside** the deep-learning features. Not a replacement for the DL signal; an auxiliary mechanism whose value is concentrated on specific problems — above all the **unbounded taxonomy** (§5.2; SOTA §1).

**Context — this already exists, aimed elsewhere.** The neuro-symbolic recipe (detect articulatory / phonological features — place, manner, voicing, vowel height/backness — on top of wav2vec2) is the established method of **Mispronunciation Detection & Diagnosis (MDD / CAPT)** ([phonological-wav2vec2 MDD](#ref-mdd-phon); [articulatory MDD](#ref-mdd-artic)). It has been pointed at *scoring a learner's pronunciation*, not at *routing*. So the tools and evidence exist; repurposing MDD-style phonological probing as an accent-**routing** signal is the novel move. L1→L2 transfer is explicit in that field — L1–L2 phonological distance is treated as the main driver of pronunciation error.

**Where the benefit genuinely arises (three places):**

1. **Unseen / low-resource accents — the strong case.** A DL classifier needs data per accent and dies off its label set. Symbolic transfer knowledge is a **zero-shot prior**: from a Spanish vowel chart you can predict Spanish-accented English will collapse /ɪ/–/iː/ (ship/sheep) and lack /æ/ *before seeing any Spanish-accented data*. Encode each candidate L1 as a predicted phonological-feature signature, match it against probes on the SSL features → interpretable zero-shot accent ID / similarity. Genuinely different from embedding-similarity (SOTA AccentFold) and complementary to it. **This is the auxiliary-for-unbounded-taxonomy idea, and the one to feature.**
2. **Interpretability → cause-matched intervention.** Phonological-feature detection says *which* contrast is off — exactly the input to the cause-matched intervention table ([§5.4](#54-interventions-mapped-to-cause--what-to-do-once-routed-incl-error-correction)). A black-box accent score can't route interventions; "merges /ɪ/–/iː/, epenthesizes s-clusters" can.
3. **Data efficiency.** Phonological features are a low-dimensional, compositional target (a handful of dimensions, not hundreds of classes), so lightweight probes on frozen SSL layers generalize where categorical models overfit — the reason MDD adopted them, since annotated L2-error data is scarce ([articulatory MDD](#ref-mdd-artic)).

**Where it will *not* help (frank cautions):**

- **On seen, well-resourced accents it likely won't beat a good DL classifier** — the bitter lesson: if you have Spanish-accented data, the SSL model already represents the merger. The wins are unseen / interpretable / efficient, not "boost the number on common accents."
- **Don't replace SSL features with hand-crafted acoustics** — formant extraction is brittle; read vowel structure off the encoder, don't recompute F1/F2 by hand.
- **Don't build the full rule repository** — an MDD finding is blunt: including too many phonological rules *drops* recognition accuracy ([E2E MDD](#ref-mdd-rules)). Concentrate on a few high-signal, L1-discriminative cues.
- **Canonical-reference trap** — dictionary "standard" pronunciations (e.g. CMUdict) are themselves accent-biased and degrade against real accented speech; use **transfer-adjusted expected** realizations, not the standard lexicon ([phonological MDD](#ref-mdd-canon)).

**Vowels first (the highest-signal entry point).** Vowel mergers/shifts are the most systematic, most L1-predictable (a vowel chart *is* the prior), most ASR-damaging (plausible-but-wrong words), and most compositional (a 2-D F1/F2 space you can reason about). A single **vowel-space distortion** probe — how shifted/compressed a speaker's vowel space is vs. a reference, and specifically whether the tense/lax high-front contrast /iː/–/ɪ/ has collapsed — is a clean, interpretable accentedness signal small enough to actually build.

**Scope for this project.** Not the core build — a full phonological-probe + transfer-prior system is a research project, not a 4–5 hour task. Its home is (a) the framing / "what I'd do with another week" argument, where the L1→L2 transfer prior is a differentiated answer to the unseen-accent problem, and (b) optionally a narrow **vowel-space demonstrator** as an interpretable "meaningful improvement" against a black-box baseline. Also a strong interview scenario topic ("marshal linguistic structure to help the model").

*Framing note (to revisit):* the working hypothesis is that this is primarily an **auxiliary mechanism for the unbounded-taxonomy problem** (zero-shot reach to unseen accents), with interpretability / cause-matched intervention as a secondary payoff — not a standalone detection method. That framing looks right; it is the thing to pressure-test if the idea gets pursued.

---

## 6. Synthesis

**Accent is a systematic, socially interpretable pattern of phonetic, phonological, and prosodic realization. In speech technology it is a distribution of acoustic and temporal cues that may cause mismatch with the default pipeline. Routing (accent-based or quality-based) is justified only when that detectable pattern predicts a better outcome from an alternate path.**

Cheat-sheet — accent property → decision:

| Accent property | Decision it forces |
|---|---|
| Distributed over many weak cues | aggregate over the utterance; learned embeddings, not rules; enough audio |
| Mostly acoustic, dies at text | trigger from SSL/encoder, not transcript; preserve metadata side-channel; cascade-vs-audio matters |
| Gradient & feature-specific | accentedness scalar / similarity over N-way argmax; hybrid trigger |
| Unbounded, fuzzy labels | embedding/similarity + quality backstop; no enumerated taxonomy |
| Predictable from L1 phonology (esp. vowels) | *optional* symbolic transfer prior as a zero-shot auxiliary signal for unseen accents (§5.6) |
| Systematic error patterns | cause-matched interventions; N-best/phonetic-grounded correction (never 1-best text) |
| Confounded with speaker/channel | speaker/corpus/channel-held-out evaluation |
| Not a deficit; labels are sensitive | measure the calibration gap; constrain use; no profiling |

---

## References

<a id="ref-yang"></a>**[Yang & Hansen 2023]** What Can an Accent Identifier Learn? Probing Phonetic and Prosodic Information in a Wav2vec2-based Accent Identification Model. Interspeech 2023 / arXiv:2306.06524. Accent-specific phoneme structure strong ~layer 9; AID fine-tuning sharpens top-2 layers; AID models lean on speaker identity. https://arxiv.org/abs/2306.06524

<a id="ref-supra"></a>**[suprasegmental probing 2024]** A layer-wise analysis of Mandarin and English suprasegmentals in SSL speech models — stress/tone/phrasal-accent probes peak in the middle third; representations are abstract, not raw F0. arXiv:2408.13678. https://arxiv.org/html/2408.13678v1

<a id="ref-orth"></a>**[layer-stratified analyses]** Orthogonality and isotropy of speaker and phonetic information in SSL speech representations (arXiv:2406.09200) — wav2vec2 phonetic info peaks in late-middle layers; HuBERT/WavLM retain it through final layers. https://arxiv.org/pdf/2406.09200

<a id="ref-ccr"></a>**[Probing CCR 2026]** Layer-wise Probing of wav2vec 2.0 and Whisper for Consonant Cluster Reduction in African American English — accent phenomena encoded as structured gradient variation in encoder layers; reduced segments retain cues to underlying stops. arXiv:2606.23948. https://arxiv.org/html/2606.23948

<a id="ref-nontimbral"></a>**[Orange 2026]** Robust Accent Identification via Voice Conversion and Non-Timbral Embeddings — speaker-invariant embeddings improve accent-ID on unseen speakers. arXiv:2604.25332. https://arxiv.org/pdf/2604.25332

<a id="ref-gh2023"></a>**[Ghorbani & Hansen 2023]** Advanced accent/dialect identification and accentedness assessment with multi-embedding models and ASR — objective accentedness scores from ASR error + AID, correlated with human perception. *J. Acoust. Soc. Am.* 155(6). https://doi.org/10.1121/10.0026235

<a id="ref-hypo"></a>**[HyPoradise 2023]** HyPoradise: An Open Baseline for Generative Speech Recognition with LLMs — N-best→transcription generative error correction (GER); ~334K pairs; surpasses re-ranking oracle. NeurIPS 2023 / arXiv:2309.15701. https://arxiv.org/pdf/2309.15701

<a id="ref-flanec"></a>**[GenSEC / FlanEC 2025]** FlanEC: Exploring Flan-T5 for Post-ASR Error Correction (SLT 2024 GenSEC challenge, HyPoradise benchmark) — LLM post-processing over N-best. arXiv:2501.12979. https://arxiv.org/pdf/2501.12979

<a id="ref-phoneticger"></a>**[phonetic GER 2025]** LLM-based Generative Error Correction for Rare Words with Synthetic Data and Phonetic Context — text-only GER ignores phonetic cues and over-corrects, reducing fidelity to spoken input; adding phonetic context helps. arXiv:2505.17410. https://arxiv.org/pdf/2505.17410

<a id="ref-csger"></a>**[Chen et al. 2023]** Generative Error Correction for Code-Switching Speech Recognition using LLMs — H2T mapping over N-best with a LoRA-adapted LLM. arXiv:2310.13013. https://arxiv.org/pdf/2310.13013

<a id="ref-mdd-phon"></a>**[phonological-wav2vec2 MDD 2026]** Using Phonological-Level Wav2Vec2 for Mandarin Mispronunciation Detection and Diagnosis — decomposes phonemes into low-level phonological components for interpretable, articulator-level diagnosis. arXiv:2606.22022. https://arxiv.org/pdf/2606.22022

<a id="ref-mdd-artic"></a>**[articulatory MDD 2023]** Multi-View Multi-Task Representation Learning for Mispronunciation Detection — articulatory features as auxiliary tasks; data-efficient gains on limited L2 data (L2-ARCTIC). arXiv:2306.01845. https://arxiv.org/pdf/2306.01845

<a id="ref-mdd-rules"></a>**[E2E MDD 2021]** End-to-End Mispronunciation Detection — notes that including too many phonological rules degrades ASR accuracy and in turn MD performance; articulatory-feature recognition as an alternative to phone symbols. arXiv:2103.03023. https://arxiv.org/pdf/2103.03023

<a id="ref-mdd-canon"></a>**[phonological MDD (canonical mismatch) 2025]** Phonological-level wav2vec2-based MDD — accuracy drops when a standard lexicon (CMUdict) is used as reference, because real accented pronunciations deviate from the dictionary standard. ScienceDirect S0167639325000640. https://www.sciencedirect.com/science/article/pii/S0167639325000640

---

*Companion: `accent-routing-sota.md` (the landscape, the 2×2, triggers, the hybrid, the cascade-vs-audio substrate).*
