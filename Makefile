.PHONY: setup smoke data features baselines probe eval report flywheel reproduce test clean \
       rescore diagnose compare ext1 ext2 ext3 ext4 ext5 features-stats reproduce-v2 \
       asr-ladder acoustic-features ext-headroom ext-composite ext-temporal ext-adapted reproduce-v3

CONFIG ?= configs/default.yaml

setup:
	uv sync

smoke:
	uv run python tests/test_smoke.py

data:
	uv run python scripts/prepare_data.py --config $(CONFIG)

asr:
	uv run python scripts/run_asr.py --config $(CONFIG)

features:
	uv run python scripts/extract_features.py --config $(CONFIG)

baselines:
	uv run python scripts/run_baselines.py --config $(CONFIG)

probe:
	uv run python scripts/train_probe.py --config $(CONFIG)

eval:
	uv run python scripts/evaluate.py --config $(CONFIG)

report:
	uv run python scripts/make_report.py --config $(CONFIG)

flywheel:
	uv run python scripts/run_flywheel.py --config $(CONFIG)

test:
	uv run python -m pytest tests/ -v

reproduce: data asr features baselines probe eval report
	@echo "Full pipeline complete. Check experiments/ for results."

clean:
	rm -rf data/asr_cache data/features_cache models/*.pt
	@echo "Caches and model artifacts cleaned."

# --- Round 2 targets ---

rescore:
	uv run python scripts/rescore_pilot.py --config $(CONFIG)

diagnose:
	uv run python scripts/diagnose.py --config $(CONFIG)

compare:
	uv run python scripts/compare.py --config $(CONFIG)

ext1:
	uv run python scripts/train_probe_ext1.py --config $(CONFIG)
	uv run python scripts/eval_ext1.py --config $(CONFIG)

ext2:
	uv run python scripts/train_learning_curve.py --config $(CONFIG)
	uv run python scripts/train_accent_classifier.py --config $(CONFIG)
	uv run python scripts/eval_ext2.py --config $(CONFIG)

ext3:
	uv run python scripts/eval_ext3.py --config $(CONFIG)

features-stats:
	uv run python scripts/extract_features_stats.py --config $(CONFIG)

ext4:
	uv run python scripts/train_probe_stats.py --config $(CONFIG)
	uv run python scripts/eval_ext4.py --config $(CONFIG)

ext5:
	uv run python scripts/train_multitask.py --config $(CONFIG)
	uv run python scripts/eval_ext5.py --config $(CONFIG)

reproduce-v2: rescore ext1 ext2 ext3 ext4 ext5 compare
	@echo "Full Round 2 pipeline complete. Check experiments/ for results."

# --- Round 3 targets ---

asr-ladder:
	uv run python scripts/run_asr_grid.py --config $(CONFIG)

acoustic-features:
	uv run python scripts/extract_acoustic_features.py --config $(CONFIG)

ext-headroom: asr-ladder
	uv run python scripts/eval_headroom_grid.py --config $(CONFIG)
	uv run python scripts/plot_headroom_sweep.py --config $(CONFIG)

ext-composite: acoustic-features
	uv run python scripts/eval_composite.py --config $(CONFIG)

ext-temporal:
	uv run python scripts/eval_temporal.py --config $(CONFIG)

ext-adapted:
	uv run python scripts/run_asr_adapted.py --config $(CONFIG)
	uv run python scripts/eval_accent_adapted.py --config $(CONFIG)

reproduce-v3: ext-headroom ext-composite ext-temporal compare
	@echo "Round 3 pipeline complete (excluding gated ext-adapted)."
