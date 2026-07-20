# Diagnosis: Stage 2 → Stage 3 (exp13-wavlm-ddp)

## What Stage 2 Set Out to Do

Move WavLM-large feature extraction off single-utterance CPU and toward
GPU readiness, in preparation for scaling extraction past the current
900-utterance dataset on a rented 2x-GPU box. Stage 1 (prior) had
already reproduced the full pipeline deterministically and fixed probe
training's missing seed. Stage 2 covered device support, correctness
verification, and batching.

## What's Verified and Closed

1. **Cache portability.** `WavLMExtractor.extract()`/`extract_stats()`
   returned tensors on whatever device they ran on; a GPU-extracted
   tensor's embedded device tag would have broken `torch.load` on any
   machine without that device (every downstream script loads with no
   `map_location`). Fixed: always `.detach().cpu()` before returning,
   regardless of extraction device. Verified as a true no-op on the
   existing 900-utterance CPU cache (900/900 `torch.equal`).

2. **Device and shard CLI plumbing.** `--device` overrides
   `configs/default.yaml`'s `features.device` per-invocation, needed to
   pin separate processes to `cuda:0`/`cuda:1` for data-parallel
   inference sharding. `--shard-index`/`--num-shards` partition the
   utterance list deterministically (`index % num_shards`) so two
   processes can extract disjoint slices into the same cache directory
   with no locking. Both default to no-op values.

3. **Device parity, unbatched.** Live CPU-vs-MPS comparison (897/900;
   the 3 most extreme-duration outliers — 536.0s/356.6s/199.5s —
   separately crash MPS's attention-bias buffer, tracked, not fixed, not
   blocking): cosine similarity min 0.99910, relative diff p99 0.77%,
   and — the number that actually matters — **downstream probe routing
   scores agree with Spearman ρ = 0.999990**. Feature-level device drift
   exists but is inconsequential to routing decisions. This is a
   CPU-vs-MPS result specifically; the real CUDA gate has not run yet.

4. **Sharding correctness.** Verified two ways: (a) logic-level, the
   exact `index % num_shards` partition used in the real scripts is
   pairwise-disjoint and its union equals the full set for 2/3/4 shards;
   (b) end-to-end, actually running both shards of a real 2-way split
   against a scratch cache directory — 900/900 files on disk, zero
   missing, zero extra, zero duplicates.

5. **Batched masking correctness.** Implemented `extract_batch`/
   `extract_stats_batch` with attention-mask-based pooling (padded
   frames excluded from both the mean and the — unbiased-estimator —
   variance), verified same-device (no device confound) against the
   independent unbatched reference. After length-bucketing (finding 1
   below), the masking-bug signature — diff correlated with how much
   padding an utterance sat next to — is gone: Pearson r(relative
   padding, relative diff) = 0.02–0.06, both statistically
   indistinguishable from zero (p=0.50, p=0.09). Holds at the extreme
   short end too (0.13s utterances, 98%+ relative padding pre-bucketing).

## The Throughput Arc (three findings, in order)

1. **Naive sequential batching was 13.5x *slower* than unbatched, on
   CPU** (342s unbatched vs 4631s batched, batch_size=16). Root cause:
   WavLM's relative-position attention bias is `O(T^2)` in the *padded
   batch's* time-frame count, set by whichever utterance in a batch is
   longest — not each utterance's own length. Batching in natural
   (unsorted) order regularly pairs short clips with long ones, forcing
   the whole batch to pay the long member's cost. **Fix:**
   length-bucketed batching (`bucket_by_duration`) — sort by duration
   before chunking. Cut the slowdown to ~3.2x (still slower than
   unbatched on CPU, but padding waste — the thing bucketing targets —
   was confirmed eliminated by the correlation test in point 5 above).

2. **Fixed batch size hits a hard OOM cliff on the duration tail.**
   Bucketing fixes padding *waste* but not peak *memory* — batch cost
   still scales with `batch_size * T^2`, and a batch_size safe for short
   utterances isn't safe for long ones. Measured: length-bucketed
   `batch_size=16` on MPS failed on 113/897 utterances (12.6%) — a sharp
   cliff at ~16.6s duration, all `MPS backend out of memory` or
   `Invalid buffer size`. Treated as a correctness-of-execution issue,
   not an optimization gap: as written, batched extraction crashed on a
   tail that's a real fraction of this dataset. **Fix:** duration-aware
   batch sizing — `bucket_by_duration`'s `max_batch_duration` parameter
   shrinks batch size for long utterances proportional to
   `(max_batch_duration / duration)^2`, down to a floor of 1. Reduced
   MPS failures from 113 to 4. The 4 remaining failures are a distinct,
   separate phenomenon (next section) — not a duration-sizing problem.
   This generalizes directly to CUDA: more VRAM moves the cliff to a
   higher duration, it does not remove it.

3. **Timing verdict on this hardware: inconclusive, and the earlier
   optimistic read didn't survive a fair measurement.** An initial fixed
   `batch_size=16` MPS run looked like a clean 6.4x win over CPU
   (164.6s vs 1055.3s) — but that number was measured on an *incomplete*
   run that bailed via OOM on 113 utterances rather than finishing them.
   Once duration-aware sizing forced MPS to actually complete the full
   897-utterance workload safely, the picture flipped: **CPU 353.8s,
   MPS 402.0s — MPS is slower than CPU** when made to finish the same
   work CPU already finishes. At batch_size≈1 for the long tail (which
   dominates wall-clock time), there's no batching parallelism left to
   exploit, and whatever advantage MPS had evaporates exactly where it
   matters. **This is not treated as "batching doesn't work"** — it's
   treated as an artifact of this Mac's small (~27GB), shared, allocator
   pool, which is a poor predictor of CUDA/A40 behavior. The timing
   question is explicitly deferred to real hardware, not resolved here.

## One Finding Documented and Deliberately Not Chased

The 4 residual MPS failures after duration-aware sizing (utterances at
93.5s, 97.7s, 124.3s, 126.0s) show a distinct signature from the OOM
cliff: each fails on a *small* final allocation (1.3GB, 19MB, 2.3GB,
24.6MB) against a pool that's *already* holding 20–24GB by utterance
#893+ of 897. Extrapolating from the known 536s/42.82GB reference point,
a single 126s utterance alone should need only ~2.4GB — comfortably
fits in 27GB fresh. The failure isn't about that utterance's size; it's
cumulative memory-pool pressure across ~900 sequential extractions in
one long-running process that MPS's allocator never released.

**Deliberately not investigated further.** This is MPS-allocator-
specific — a small, shared, comparatively immature memory-management
implementation. Chasing it here would be measuring this machine, not
predicting CUDA's more mature caching allocator. Documented so it's not
rediscovered from scratch; re-check on CUDA if (and only if) a similar
pattern appears there.

## Also Confirmed, Also Deferred

The `mismatched key_padding_mask and attn_mask` deprecation warning
(WavLM's own relative-position bias combined with the padding
`attention_mask`, likely forcing an unfused/generic attention kernel)
is confirmed **general** — fires identically on CPU and MPS under
forced `PYTHONWARNINGS=always` (an initial CPU-only-seeming appearance
was Python's per-process warning dedup, not a device difference; caught
and corrected mid-investigation). It does not appear to be *blocking*
batching's benefit — MPS still substantially outpaced CPU on the
subset it could complete even while carrying this inefficiency. Status:
**deferred, not fixed** — a known optimization to attempt on CUDA,
where the actual gain can be measured directly rather than inferred,
or skipped if throughput is already sufficient without it.

## The Decision Carried Into Stage 3

Two extraction paths, not one:

- **Sharded-unbatched — the default.** `--shard-index`/`--num-shards`,
  no batching. Embarrassingly parallel, cannot OOM (no batch dimension
  to blow up), needs no bucketing or masking logic, and already
  completes 900 utterances in ~350s on CPU alone. This is the robust
  baseline for the A40 box.
- **Batched-bucketed (duration-aware) — the experiment.** Carries real,
  verified correctness (masking, sharding composition) and a real,
  verified reliability fix (OOM cliff), but its throughput case is
  *unproven* — the only clean measurement available (this Mac) argues
  against it, for reasons believed to be hardware-specific. To be
  benchmarked against the sharded-unbatched default on real CUDA
  hardware. If it wins clearly, use it; if it's a wash, ship the simpler
  path. **Batching is the experiment, not the assumption** going into
  Stage 3.
