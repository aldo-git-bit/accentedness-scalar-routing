# Batching efficiency findings (exp13-wavlm-ddp)

Three related findings from implementing and verifying batched WavLM
extraction, in the order they were discovered. All measured on the same
897-utterance set (900 minus the 3 most extreme-duration outliers,
536.0s/356.6s/199.5s, tracked separately).

## 1. Naive sequential batching was 13.5x slower than unbatched, on CPU

### The finding

While implementing batched WavLM extraction on the `exp13-wavlm-ddp`
branch, the first working version — batching utterances in whatever
order they happened to load in (natural train/val/test pickle order,
not sorted by duration) — was dramatically **slower** than no batching
at all, on the same 897-utterance set, same CPU, same model:

| | Unbatched | Batched (naive, batch_size=16) | Ratio |
|---|---|---|---|
| Mean-pooled extraction | 342.0s | 4631.4s | 13.5x slower |
| Mean+std extraction | 353.2s | 4692.9s | 13.3x slower |

Batching, as implemented, made throughput *worse* by over an order of
magnitude, despite doing the same 897 utterances in 57 forward-pass
calls instead of 897.

### Root cause

WavLM's relative-position attention bias is `O(T^2)` in the padded
batch's time-frame count `T` — and `T` is set by whichever utterance in
a batch is *longest*, not by each utterance's own length. This
dataset's utterance durations are long-tailed: median ~3s, but a
meaningful tail runs out to 75s, 97s, 126s, and beyond (the most
extreme, at 536s/356s/199s, separately crash MPS's memory allocator on
this hardware — see `verify_features_device_parity.py`'s findings).

Batching in natural (unsorted) order means a batch can easily contain
one ~126s utterance sitting next to fifteen ~3s utterances. Every one
of those short clips then gets padded and forced through the transformer
at ~126s-equivalent length, multiplying compute for the *entire batch*
by whatever its longest member happens to be. With a long-tailed
duration distribution, most batches end up dominated by whichever tail
utterance they happened to land next to — the waste isn't an edge case,
it's close to the typical case.

### The fix

Length-bucketed batching: sort utterances by duration before chunking
into batches (`accentedness_routing.features.batching.bucket_by_duration`),
so each batch's members are close in length and padding waste stays
small. This is a change to batch *construction* only — the masking math
in `WavLMExtractor` (verified correct in the prior round) is untouched.

### Why this is worth keeping around

This is a clean, reproducible, first-person demonstration of a general
lesson in variable-length-sequence batching: naive batching of
heterogeneous-length data isn't just "less optimal," it can be actively
counterproductive, and the failure mode (padding waste dominated by the
longest member of an arbitrary batch) generalizes well beyond this
project — it will matter again anywhere batched inference meets
long-tailed input lengths, including the eventual GPU/A100 scaling work
this branch exists to prepare for.

## 2. Fixed batch size hits a hard OOM cliff on the long duration tail

### The finding

Length-bucketing (finding 1's fix) removes padding *waste*, but a fixed
`batch_size` still doesn't work across this dataset's full duration
range on memory-constrained hardware. Running length-bucketed
`batch_size=16` extraction CPU-vs-MPS on all 897 utterances: CPU
completed all 897/897. MPS failed on **113/897 (12.6%)** — every
utterance in the batch construction ≥16.6s duration, all with
`MPS backend out of memory` or `Invalid buffer size` errors against
MPS's ~27GB shared memory pool. The failure boundary is sharp, not
gradual: batches under ~15s at batch_size=16 succeeded cleanly; batches
at or above ~16.6s failed outright.

This is a correctness-of-execution issue, not just a missed
optimization: as implemented at the time, batched extraction crashes on
a long tail that makes up more than an eighth of this dataset.

### Root cause

Same `O(T^2)` attention-bias scaling as finding 1, now hitting a hard
memory ceiling instead of just wasting compute. Batch peak memory
scales with `batch_size * T^2`; a fixed `batch_size` that's safe for
short utterances is not safe for long ones. This generalizes directly
to CUDA/A100: more VRAM moves the failure threshold to a higher
duration, it does not remove the threshold. A production script that
assumes one fixed batch size will eventually hit this on any dataset
with a long enough tail, on any device.

### The fix

Duration-aware batch sizing: `bucket_by_duration`'s optional
`max_batch_duration` parameter shrinks the batch size for utterances
longer than that threshold, proportional to `(max_batch_duration /
duration)^2` (matching the quadratic cost scaling), down to a floor of
1. Calibrated from this run's MPS boundary (`max_batch_duration=15.0`
seconds at `batch_size=16` is the last-known-good point observed here)
— this number is MPS-27GB-specific and should be recalibrated against
real numbers once running on the A100, not trusted as a universal
constant. Default behavior (`max_batch_duration=None`) is unchanged, so
this is strictly opt-in.

## 3. The mask-dispatch deprecation warning is confirmed general, not CPU-specific

### The finding

Batched extraction triggers a PyTorch warning: `Support for mismatched
key_padding_mask and attn_mask is deprecated` (from
`torch/nn/functional.py:6441`, `transformers==5.13.0`'s WavLM attention
implementation combining its own relative-position bias with the
padding `attention_mask` we pass for batching). In an initial CPU-then-MPS
comparison run, the warning printed once during the CPU pass and never
appeared during the MPS pass — which looked like it might mean MPS
avoids this code path. It doesn't: Python's default warning filter
shows each unique (message, file, line) combination only once **per
process**, and both devices fire the identical warning from the same
line, so the apparent CPU-only occurrence was a dedup artifact, not a
device difference. Confirmed by forcing `PYTHONWARNINGS=always` on a
small isolated batch run on each device independently: the warning
fires identically on both.

### Why it doesn't block the batching win

Despite this shared code path (likely an unfused/generic attention
kernel rather than a fused optimized one, though not directly measured
in isolation), MPS still completed the 784/897 utterances it could fit
in memory 6.4x faster than CPU completed all 897/897. Whatever the
mask-mismatch costs, it isn't preventing a real speedup from batching
on real accelerator hardware — it's an efficiency layer on top of an
already-working improvement, not a prerequisite for one.

### Status: deferred

Not fixed. Documented as a known optimization to attempt on the A100
(unify the two masks so PyTorch's fused kernel path can engage), where
the actual speed gain can be measured directly rather than inferred —
or skip it entirely if A100 throughput is already sufficient without it.
