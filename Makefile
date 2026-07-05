.PHONY: setup smoke data features baselines probe eval report reproduce test clean

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

test:
	uv run pytest tests/ -v

reproduce: data asr features baselines probe eval report
	@echo "Full pipeline complete. Check experiments/ for results."

clean:
	rm -rf data/asr_cache data/features_cache models/*.pt
	@echo "Caches and model artifacts cleaned."
