# Diagnosis: Round 2 → Round 3

## The Binding Constraint Is Headroom

Round 2 established that the **escalation headroom** — the gap between default and careful model performance — is the binding constraint on routing value, not the trigger quality.

Key findings from Round 2:

1. **Oracle area-vs-random is only 0.031.** Even a perfect trigger (oracle) improves net WER by only 3.1 percentage points compared to random escalation. This is the *ceiling* for any learned trigger.

2. **Confidence captures 47% of headroom for free.** The confidence trigger (avg_logprob from the default model) achieves area-vs-random ≈ 0.015, roughly half the oracle ceiling — with zero additional model cost or feature extraction.

3. **The remaining gap is smaller than CI width.** The residual between confidence and oracle is ~0.016, which is comparable to the 95% CI width (≈±0.012). Any learned trigger must demonstrate improvement against this noisy residual.

4. **Learned probes (WavLM scalar, gain-target, multitask) cluster near confidence.** Despite using 25-layer WavLM representations, the best learned triggers (multitask λ=0.1) only marginally improve over confidence, and the difference is not statistically significant.

## Implications

The Round 2 result is *not* "learned triggers don't work." It is: **when the model gap is narrow (small → large-v3), there isn't enough signal for any trigger to exploit.** Confidence captures the easy wins (high-WER utterances tend to have low confidence), and the remaining cases where escalation helps but confidence is uninformative are too few to move the aggregate metric.

## Round 3 Strategy: Manipulate Headroom

Rather than continuing to refine triggers against a narrow ceiling, Round 3 tests the hypothesis that **wider model gaps create exploitable headroom**:

1. **Headroom grid (Phase 1):** Test 6 model pairings ({tiny, base, small} × {turbo, large-v3}). Wider gaps (e.g., tiny → large-v3) should produce larger oracle area and more utterances with positive escalation gain.

2. **Composite signals (Phase 2):** Combine confidence with acoustic features (duration, silence ratio, speaking rate) and the champion learned trigger via logistic regression. Test whether the composite exceeds confidence-alone specifically in the wider-gap regimes.

3. **Temporal features (Phase 3):** Add per-layer WavLM std (temporal variability) to the combiner. Test whether this helps on the "confidently hallucinated" subset where confidence is blind.

4. **Accent-adapted careful path (Phase 4, stretch):** Test whether an Indian-accent finetuned model expands headroom on the Indian English subset.

## A Negative Result Is Legitimate

If Round 3 finds that confidence captures most of the headroom even at wider gaps, this is a valid finding: **use free confidence until the model gap justifies learned routing.** The headroom sweep figure (Phase 1) quantifies exactly where that boundary lies.

## Three Levers

| Lever | Mechanism | Phase |
|-------|-----------|-------|
| Widen the gap | Use weaker default models (tiny, base) paired with stronger careful models | 1 |
| Composite signals | Combine multiple cheap signals via logistic regression | 2–3 |
| Accent-adapted path | Specialized careful model for specific accent groups | 4 |
