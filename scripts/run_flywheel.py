"""Run flywheel analysis: drift detection and hard-case mining."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.flywheel.drift import simulate_accent_shift
from accentedness_routing.flywheel.mining import mine_hard_cases
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-03-flywheel"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load test utterances
    with open(data_dir / "test_utterances.pkl", "rb") as f:
        test_utts = pickle.load(f)

    # Load ASR results
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    default_wers = {}
    careful_wers = {}
    accent_map = {}

    for utt in test_utts:
        uid = utt.utterance_id
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d is None or c is None:
            continue
        default_wers[uid] = d["wer"]
        careful_wers[uid] = c["wer"]
        accent_map[uid] = utt.accent

    utt_ids = list(default_wers.keys())

    # Load probe
    model_path = Path(cfg["probe"]["model_path"])
    if not model_path.exists():
        print("No probe model found. Run `make probe` first.")
        return

    checkpoint = torch.load(model_path, weights_only=False)
    probe = AccentednessProbe(
        num_layers=cfg["features"]["num_layers"],
        hidden_dim=cfg["features"]["hidden_dim"],
        probe_dim=cfg["probe"]["hidden_dim"],
        dropout=cfg["probe"]["dropout"],
    )
    probe.load_state_dict(checkpoint["model_state_dict"])
    calibration = checkpoint["calibration"]

    # Load features
    features_dir = Path(cfg["features"]["cache_dir"])
    test_features = {}
    for uid in utt_ids:
        fp = features_dir / f"{uid}.pt"
        if fp.exists():
            test_features[uid] = torch.load(fp, weights_only=True)

    trigger = ScalarProbeTrigger(probe, test_features, calibration)
    scores = {uid: trigger.score(uid) for uid in utt_ids if uid in test_features}

    # Group scores by accent
    scores_by_accent = defaultdict(list)
    for uid, score in scores.items():
        scores_by_accent[accent_map[uid]].append(score)

    # Drift detection
    print("=== Drift Detection ===")
    drift_result = simulate_accent_shift(dict(scores_by_accent))
    print(f"  KS statistic: {drift_result['ks_statistic']:.4f}")
    print(f"  p-value: {drift_result['p_value']:.6f}")
    print(f"  Detected: {drift_result['detected']}")

    # Hard-case mining
    print("\n=== Hard-Case Mining ===")
    hard_cases = mine_hard_cases(
        list(scores.keys()), scores, default_wers, careful_wers, top_k=10
    )

    print("\nHighest scalar scores:")
    for case in hard_cases["highest_score"][:5]:
        print(f"  {case['utterance_id']}: score={case['score']:.3f} "
              f"default_wer={case['default_wer']:.3f} careful_wer={case['careful_wer']:.3f}")

    print("\nHighest misrouting cost:")
    for case in hard_cases["highest_misrouting_cost"][:5]:
        print(f"  {case['utterance_id']}: cost={case['cost']:.3f} "
              f"default_wer={case['default_wer']:.3f} careful_wer={case['careful_wer']:.3f}")

    # Save results
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return obj.item()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results = {
        "drift": drift_result,
        "hard_cases": hard_cases,
    }
    with open(exp_dir / "flywheel_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nSaved to {exp_dir}")


if __name__ == "__main__":
    main()
