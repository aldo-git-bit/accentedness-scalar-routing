# Checkpoint provenance

`probe.pt` is, as of the `exp13-wavlm-ddp` branch, the **reseeded** checkpoint
(trained after `torch.manual_seed` + cuDNN determinism knobs were added to
`triggers/train_probe.py::train_probe()`), not the original checkpoint that
shipped with the repo before this branch. Routing numbers differ slightly
from the original (see `experiments/EXP-01-baselines/CONTAMINATION-NOTE.md`
history / conversation log for the before/after diff) but are within normal
run-to-run variance and do not regress. `models/*.pt` is gitignored, so this
note is the only durable record of which checkpoint is "current."

`probe_gain.pt` and `probe_capped_wer.pt` (from `scripts/train_probe_ext1.py`,
EXP-04) are, as of the same branch, **explicitly seeded via `cfg["seed"]`**
(previously relied on `train_probe()`'s default parameter value happening to
match config, not wired through) and **verified bit-identical across two
independent seeded runs** — every state-dict tensor, calibration, and
training history match exactly. `probe_gain.pt`'s target is
`escalation_gain(wer_default, wer_careful)` (capped internally via
`eval_common.cap_wer`) — this is the ground-truth checkpoint EXP-13's
frozen-encoder control validates against, not `probe.pt` (whose target is
raw uncapped default-model WER, the pilot's contaminated target).

The other probe-training siblings (`train_multitask.py`, `train_probe_stats.py`,
`train_learning_curve.py`, `train_accent_classifier.py`) still only fall
through to `train_probe()`'s default seed rather than reading `cfg["seed"]`
explicitly — flagged, not fixed, out of scope for the current work.
