# Checkpoint provenance

`probe.pt` is, as of the `exp13-wavlm-ddp` branch, the **reseeded** checkpoint
(trained after `torch.manual_seed` + cuDNN determinism knobs were added to
`triggers/train_probe.py::train_probe()`), not the original checkpoint that
shipped with the repo before this branch. Routing numbers differ slightly
from the original (see `experiments/EXP-01-baselines/CONTAMINATION-NOTE.md`
history / conversation log for the before/after diff) but are within normal
run-to-run variance and do not regress. `models/*.pt` is gitignored, so this
note is the only durable record of which checkpoint is "current."
