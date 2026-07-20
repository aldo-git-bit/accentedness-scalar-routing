"""Length-bucketed batch construction for WavLM extraction.

Naive sequential batching (chunking utterances in whatever order they
happen to load in) pads every batch to its longest member's duration.
With a long-tailed duration distribution this makes batched extraction
dramatically *slower* than unbatched (see
docs/DIAGNOSIS-batching-padding-waste.md: 13.5x slower, measured).
Sorting by duration before chunking keeps each batch's members close in
length, so padding waste stays small.

A fixed batch_size still isn't safe across a long-tailed duration range,
though: WavLM's O(T^2) attention-bias memory cost means a batch_size
that's fine for short utterances can OOM on long ones (measured: 12.6%
of a 897-utterance set failed on MPS at batch_size=16 for utterances
past ~16s — see the same doc's finding 2). The optional
max_batch_duration parameter shrinks batch size for long utterances to
avoid that.

This module only decides which utterances share a batch — it has no
opinion on how a batch is processed (that's WavLMExtractor's job) or on
which utterances a caller passes in (sharding, cache-filtering, etc.
happen upstream, in the caller).
"""

from __future__ import annotations


def bucket_by_duration(
    utterances: list, batch_size: int, max_batch_duration: float | None = None
) -> list[list]:
    """Sort utterances by duration, then chunk into batches.

    Only reorders *processing* — output is still keyed per
    utterance-id by the caller, so bucketing is invisible to the cache.
    Caller should pass in an already shard-filtered, already
    cache-filtered list, so bucketing happens within one shard's slice
    of work rather than across the full dataset.

    Args:
        utterances: utterances to batch (already shard/cache-filtered).
        batch_size: batch size for utterances at or under
            max_batch_duration.
        max_batch_duration: if None (default), every batch has exactly
            batch_size utterances (or fewer for the last batch) —
            duration-blind, matches pre-existing behavior exactly. If
            set, utterances longer than this get a proportionally
            smaller batch size — batch_size * (max_batch_duration /
            duration)^2, floored at 1 — matching how peak memory scales
            with batch_size * duration^2. This threshold is
            hardware/memory-specific (calibrated per-device from
            observed OOM boundaries, not a universal constant); recalibrate
            it for a different device rather than assuming the value that
            was safe on one machine is safe on another.
    """
    sorted_utts = sorted(utterances, key=lambda u: len(u.audio) / u.sample_rate)

    if max_batch_duration is None:
        return [sorted_utts[i : i + batch_size] for i in range(0, len(sorted_utts), batch_size)]

    batches: list[list] = []
    i = 0
    n = len(sorted_utts)
    while i < n:
        dur = len(sorted_utts[i].audio) / sorted_utts[i].sample_rate
        if dur <= max_batch_duration:
            this_batch_size = batch_size
        else:
            this_batch_size = max(1, int(batch_size * (max_batch_duration / dur) ** 2))
        batches.append(sorted_utts[i : i + this_batch_size])
        i += this_batch_size
    return batches
