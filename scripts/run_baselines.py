"""Compute operating curves for all baseline triggers."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.routing.metrics import compute_summary
from accentedness_routing.routing.router import compute_operating_curve
from accentedness_routing.triggers.commonaccent import ArgmaxAccentTrigger
from accentedness_routing.triggers.confidence import ConfidenceTrigger
from accentedness_routing.triggers.oracle import OracleTrigger
from accentedness_routing.triggers.random_trigger import RandomTrigger


def load_wer_data(utterances, cfg) -> tuple[dict, dict, dict, dict]:
    """Load cached ASR results and return WER dicts + metadata."""
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    default_wers = {}
    careful_wers = {}
    logprobs = {}
    accent_map = {}

    for utt in utterances:
        uid = utt.utterance_id

        d_result = load_cached(cache_dir, default_model, uid)
        c_result = load_cached(cache_dir, careful_model, uid)

        if d_result is None or c_result is None:
            print(f"Warning: missing ASR cache for {uid}, skipping")
            continue

        default_wers[uid] = d_result["wer"]
        careful_wers[uid] = c_result["wer"]
        logprobs[uid] = d_result["avg_logprob"]
        accent_map[uid] = utt.accent

    return default_wers, careful_wers, logprobs, accent_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load test split utterances
    data_dir = Path("data")
    with open(data_dir / "test_utterances.pkl", "rb") as f:
        test_utts = pickle.load(f)
    print(f"Loaded {len(test_utts)} test utterances")

    default_wers, careful_wers, logprobs, accent_map = load_wer_data(test_utts, cfg)
    utt_ids = list(default_wers.keys())
    print(f"  {len(utt_ids)} utterances with complete ASR results")

    # Summary stats
    mean_default = sum(default_wers.values()) / len(default_wers)
    mean_careful = sum(careful_wers.values()) / len(careful_wers)
    print(f"  Mean default WER: {mean_default:.4f}")
    print(f"  Mean careful WER: {mean_careful:.4f}")

    # Compute per-accent mean WER for argmax baseline
    accent_wer_sums = defaultdict(float)
    accent_counts = defaultdict(int)
    for uid in utt_ids:
        accent_wer_sums[accent_map[uid]] += default_wers[uid]
        accent_counts[accent_map[uid]] += 1
    accent_mean_wers = {
        acc: accent_wer_sums[acc] / accent_counts[acc] for acc in accent_wer_sums
    }

    # Build triggers
    triggers = [
        OracleTrigger(default_wers, careful_wers),
        RandomTrigger(seed=cfg.get("seed", 42)),
        ConfidenceTrigger(logprobs),
        ArgmaxAccentTrigger(accent_map, accent_mean_wers),
    ]

    # Compute curves
    num_thresholds = cfg["routing"]["num_thresholds"]
    curves = {}
    for trigger in triggers:
        print(f"\nComputing curve: {trigger.name}")
        curve = compute_operating_curve(
            trigger, utt_ids, default_wers, careful_wers, num_thresholds
        )
        curves[trigger.name] = curve

    # Also add floor lines
    curves["default_only"] = {
        "trigger_name": "default_only",
        "escalation_rates": [0.0],
        "net_wers": [mean_default],
        "thresholds": [1.0],
    }
    curves["careful_only"] = {
        "trigger_name": "careful_only",
        "escalation_rates": [1.0],
        "net_wers": [mean_careful],
        "thresholds": [0.0],
    }

    # Compute summaries
    budget_points = cfg["routing"]["budget_points"]
    random_curve = curves["random"]

    summaries = {}
    for name, curve in curves.items():
        if name in ("default_only", "careful_only"):
            continue
        summary = compute_summary(curve, random_curve, budget_points)
        summaries[name] = summary
        print(f"\n{name}:")
        for k, v in summary.items():
            if k != "trigger_name":
                print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

    # Save results
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-01-baselines"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    with open(exp_dir / "summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nSaved to {exp_dir}")


if __name__ == "__main__":
    main()
