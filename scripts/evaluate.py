"""Full evaluation: curves, per-accent analysis, diagnostics, plots."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.curves import compute_all_curves
from accentedness_routing.eval.diagnostics import error_profile, speaker_leakage_check
from accentedness_routing.eval.plots import (
    plot_layer_weights,
    plot_operating_curves,
    plot_per_accent_delta,
    plot_score_distributions,
)
from accentedness_routing.eval.slicing import per_accent_analysis
from accentedness_routing.routing.metrics import dominates
from accentedness_routing.triggers.commonaccent import ArgmaxAccentTrigger
from accentedness_routing.triggers.confidence import ConfidenceTrigger
from accentedness_routing.triggers.oracle import OracleTrigger
from accentedness_routing.triggers.random_trigger import RandomTrigger
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-02-scalar-vs-baselines"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load test utterances
    with open(data_dir / "test_utterances.pkl", "rb") as f:
        test_utts = pickle.load(f)
    print(f"Loaded {len(test_utts)} test utterances")

    # Load ASR results
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    default_wers = {}
    careful_wers = {}
    logprobs = {}
    accent_map = {}
    speaker_map = {}

    for utt in test_utts:
        uid = utt.utterance_id
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d is None or c is None:
            continue
        default_wers[uid] = d["wer"]
        careful_wers[uid] = c["wer"]
        logprobs[uid] = d["avg_logprob"]
        accent_map[uid] = utt.accent
        speaker_map[uid] = utt.speaker

    utt_ids = list(default_wers.keys())
    print(f"  {len(utt_ids)} utterances with ASR results")

    # Per-accent mean WER for argmax trigger
    accent_wer_sums = defaultdict(float)
    accent_counts = defaultdict(int)
    for uid in utt_ids:
        accent_wer_sums[accent_map[uid]] += default_wers[uid]
        accent_counts[accent_map[uid]] += 1
    accent_mean_wers = {a: accent_wer_sums[a] / accent_counts[a] for a in accent_wer_sums}

    # Build triggers
    triggers = [
        OracleTrigger(default_wers, careful_wers),
        RandomTrigger(seed=cfg.get("seed", 42)),
        ConfidenceTrigger(logprobs),
        ArgmaxAccentTrigger(accent_map, accent_mean_wers),
    ]

    # Load probe if available
    model_path = Path(cfg["probe"]["model_path"])
    if model_path.exists():
        checkpoint = torch.load(model_path, weights_only=False)
        probe = AccentednessProbe(
            num_layers=cfg["features"]["num_layers"],
            hidden_dim=cfg["features"]["hidden_dim"],
            probe_dim=cfg["probe"]["hidden_dim"],
            dropout=cfg["probe"]["dropout"],
        )
        probe.load_state_dict(checkpoint["model_state_dict"])
        calibration = checkpoint["calibration"]

        # Load features for test set
        features_dir = Path(cfg["features"]["cache_dir"])
        test_features = {}
        for uid in utt_ids:
            fp = features_dir / f"{uid}.pt"
            if fp.exists():
                test_features[uid] = torch.load(fp, weights_only=True)

        if test_features:
            probe_trigger = ScalarProbeTrigger(probe, test_features, calibration)
            triggers.append(probe_trigger)
            print(f"  Loaded probe with {len(test_features)} test features")

            # Plot layer weights
            layer_weights = checkpoint.get("layer_weights", [])
            if layer_weights:
                plot_layer_weights(layer_weights, str(fig_dir / "layer_weights.png"))

            # Score distributions by accent
            scores_by_accent = defaultdict(list)
            for uid in utt_ids:
                if uid in test_features:
                    scores_by_accent[accent_map[uid]].append(probe_trigger.score(uid))
            plot_score_distributions(
                dict(scores_by_accent), str(fig_dir / "score_distributions.png")
            )
    else:
        print("  No probe model found, skipping scalar probe trigger")

    # Compute all curves
    print("\nComputing operating curves...")
    result = compute_all_curves(
        triggers, utt_ids, default_wers, careful_wers,
        num_thresholds=cfg["routing"]["num_thresholds"],
        budget_points=cfg["routing"]["budget_points"],
    )

    # Add floor lines
    mean_default = sum(default_wers.values()) / len(default_wers)
    mean_careful = sum(careful_wers.values()) / len(careful_wers)
    result["curves"]["default_only"] = {
        "trigger_name": "default_only",
        "escalation_rates": [0.0],
        "net_wers": [mean_default],
        "thresholds": [1.0],
    }
    result["curves"]["careful_only"] = {
        "trigger_name": "careful_only",
        "escalation_rates": [1.0],
        "net_wers": [mean_careful],
        "thresholds": [0.0],
    }

    # Print summaries
    print("\n=== Summary Metrics ===")
    for name, summary in result["summaries"].items():
        print(f"\n{name}:")
        for k, v in summary.items():
            if k != "trigger_name" and v is not None:
                print(f"  {k}: {v:.4f}")

    # Dominance check
    if "scalar_probe" in result["curves"] and "argmax_accent" in result["curves"]:
        dom = dominates(result["curves"]["scalar_probe"], result["curves"]["argmax_accent"])
        print(f"\n{'SCALAR DOMINATES ARGMAX' if dom else 'Scalar does NOT dominate argmax'}")

    # Plot operating curves
    plot_operating_curves(result["curves"], str(fig_dir / "operating_curves.png"))

    # Per-accent analysis
    trigger_scores = {}
    for t in triggers:
        trigger_scores[t.name] = {uid: t.score(uid) for uid in utt_ids}

    per_accent = per_accent_analysis(
        utt_ids, accent_map, default_wers, careful_wers, trigger_scores
    )
    plot_per_accent_delta(per_accent, str(fig_dir / "per_accent_wer.png"))

    # Diagnostics
    if "scalar_probe" in trigger_scores:
        leakage = speaker_leakage_check(
            utt_ids, trigger_scores["scalar_probe"], speaker_map, accent_map
        )
        print(f"\nSpeaker leakage check:")
        print(f"  MI(score, speaker) = {leakage['mi_score_speaker']:.4f}")
        print(f"  MI(score, accent)  = {leakage['mi_score_accent']:.4f}")
        print(f"  Ratio = {leakage['ratio_speaker_to_accent']:.2f}")
        result["leakage"] = leakage

    profiles = error_profile(utt_ids, default_wers, careful_wers, accent_map)
    result["error_profiles"] = profiles
    result["per_accent"] = per_accent

    # Save everything
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nAll results saved to {exp_dir}")


if __name__ == "__main__":
    main()
