"""Extension 4 evaluation: Stats pooling ablation.

Scores mean+std, mean-only, std-only probes through eval_common.
Focus analysis on catastrophic subset (capped escalation gain > 0.5).

Produces: experiments/EXP-07-extension4-stats-pooling/
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import (
    bootstrap,
    decision_scorecard,
    escalation_gain,
    operating_curve,
    paired_bootstrap,
    summarize,
)
from accentedness_routing.triggers.random_trigger import RandomTrigger
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger


def load_probe_scores(model_path, cfg, test_features, name):
    """Load a probe and compute test scores."""
    if not model_path.exists():
        print(f"  {name}: model not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, weights_only=False)
    input_dim = checkpoint["config"].get("hidden_dim_input",
                                         cfg["features"]["hidden_dim"])
    probe = AccentednessProbe(
        num_layers=cfg["features"]["num_layers"],
        hidden_dim=input_dim,
        probe_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    )
    probe.load_state_dict(checkpoint["model_state_dict"])
    trigger = ScalarProbeTrigger(probe, test_features, checkpoint["calibration"])
    return {uid: trigger.score(uid) for uid in test_features}


def main():
    parser = argparse.ArgumentParser(description="Extension 4: Stats pooling evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-07-extension4-stats-pooling"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load test utterances
    with open(data_dir / "test_utterances.pkl", "rb") as f:
        test_utts = pickle.load(f)

    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    default_wers: dict[str, float] = {}
    careful_wers: dict[str, float] = {}

    for utt in test_utts:
        uid = utt.utterance_id
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d is None or c is None:
            continue
        default_wers[uid] = d["wer"]
        careful_wers[uid] = c["wer"]

    utt_ids = sorted(default_wers.keys())
    gains = {uid: escalation_gain(default_wers[uid], careful_wers[uid]) for uid in utt_ids}

    # Load features
    stats_dir = Path(cfg["features_stats"]["cache_dir"])
    mean_dir = Path(cfg["features"]["cache_dir"])

    stats_features = {}
    mean_features = {}
    std_features = {}

    for uid in utt_ids:
        sp = stats_dir / f"{uid}.pt"
        mp = mean_dir / f"{uid}.pt"
        if sp.exists():
            sf = torch.load(sp, weights_only=True)
            stats_features[uid] = sf
            std_features[uid] = sf[:, 1024:]  # std columns only
        if mp.exists():
            mean_features[uid] = torch.load(mp, weights_only=True)

    print(f"Stats features: {len(stats_features)}, Mean features: {len(mean_features)}")

    models_dir = Path("models")
    random_trigger = RandomTrigger(seed=cfg.get("seed", 42))
    random_scores = {uid: random_trigger.score(uid) for uid in utt_ids}

    # Load all probe variants
    triggers: dict[str, dict[str, float]] = {"random": random_scores}

    if stats_features:
        scores = load_probe_scores(
            models_dir / "probe_mean_std.pt", cfg, stats_features, "mean_std")
        if scores:
            triggers["probe_mean_std"] = scores

    if mean_features:
        scores = load_probe_scores(
            models_dir / "probe_mean_only.pt", cfg, mean_features, "mean_only")
        if scores:
            triggers["probe_mean_only"] = scores

    if std_features:
        scores = load_probe_scores(
            models_dir / "probe_std_only.pt", cfg, std_features, "std_only")
        if scores:
            triggers["probe_std_only"] = scores

    # Score all
    results: dict = {"triggers": {}}

    for trig_name, trig_scores in triggers.items():
        print(f"\nScoring: {trig_name}")
        curve = operating_curve(trig_scores, default_wers, careful_wers)
        random_curve = operating_curve(random_scores, default_wers, careful_wers)
        summary = summarize(curve, random_curve)

        scorecard_0 = decision_scorecard(trig_scores, gains, tau=0.0)
        scorecard_05 = decision_scorecard(trig_scores, gains, tau=0.05)

        boot = bootstrap(
            trig_scores, default_wers, careful_wers,
            n=args.bootstrap_n, seed=cfg.get("seed", 42),
            random_scores=random_scores,
        )

        results["triggers"][trig_name] = {
            "summary": summary,
            "scorecard_tau_0.00": scorecard_0,
            "scorecard_tau_0.05": scorecard_05,
            "bootstrap_ci": boot["ci"],
            "curve_bands": boot["curve_bands"],
        }

        for k, v in summary.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

    # Paired bootstrap: mean+std vs mean-only
    if "probe_mean_std" in triggers and "probe_mean_only" in triggers:
        print("\nPaired: mean+std vs mean-only")
        paired = paired_bootstrap(
            triggers["probe_mean_std"], triggers["probe_mean_only"],
            default_wers, careful_wers,
            n=args.bootstrap_n, seed=cfg.get("seed", 42),
            random_scores=random_scores,
        )
        results["paired_mean_std_vs_mean_only"] = paired

    # Catastrophic subset analysis (gain > 0.5)
    catastrophic_uids = [uid for uid in utt_ids if gains[uid] > 0.5]
    if catastrophic_uids and len(catastrophic_uids) >= 5:
        print(f"\nCatastrophic subset: {len(catastrophic_uids)} utterances (gain > 0.5)")
        results["catastrophic_subset"] = {
            "n_utterances": len(catastrophic_uids),
        }
        for trig_name, trig_scores in triggers.items():
            if trig_name == "random":
                continue
            subset_scores = {uid: trig_scores[uid] for uid in catastrophic_uids
                             if uid in trig_scores}
            subset_gains = {uid: gains[uid] for uid in catastrophic_uids}
            if subset_scores:
                sc = decision_scorecard(subset_scores, subset_gains, tau=0.5)
                results["catastrophic_subset"][trig_name] = sc
                print(f"  {trig_name}: AUC={sc.get('auc')}, Pearson r={sc.get('pearson_r')}")

    # Save
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
