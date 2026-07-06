"""Extension 5 evaluation: Multi-task probe lambda sweep.

Scores regression head for each lambda through eval_common.
Computes MI(score, accent) and MI(score, speaker) per lambda.
Paired bootstrap: best lambda vs champion single-task probe.

Produces: experiments/EXP-08-extension5-multitask/
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
from accentedness_routing.eval.diagnostics import speaker_leakage_check
from accentedness_routing.triggers.multitask_probe import MultiTaskProbe
from accentedness_routing.triggers.random_trigger import RandomTrigger
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def main():
    parser = argparse.ArgumentParser(description="Extension 5: Multi-task evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-08-extension5-multitask"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load test utterances
    with open(data_dir / "test_utterances.pkl", "rb") as f:
        test_utts = pickle.load(f)

    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    default_wers: dict[str, float] = {}
    careful_wers: dict[str, float] = {}
    accent_map: dict[str, str] = {}
    speaker_map: dict[str, str] = {}

    for utt in test_utts:
        uid = utt.utterance_id
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d is None or c is None:
            continue
        default_wers[uid] = d["wer"]
        careful_wers[uid] = c["wer"]
        accent_map[uid] = utt.accent
        speaker_map[uid] = utt.speaker

    utt_ids = sorted(default_wers.keys())
    gains = {uid: escalation_gain(default_wers[uid], careful_wers[uid]) for uid in utt_ids}

    # Load features
    features_dir = Path(cfg["features"]["cache_dir"])
    test_features = {}
    for uid in utt_ids:
        fp = features_dir / f"{uid}.pt"
        if fp.exists():
            test_features[uid] = torch.load(fp, weights_only=True)

    print(f"Test: {len(utt_ids)} utterances, {len(test_features)} with features")

    models_dir = Path("models")
    random_trigger = RandomTrigger(seed=cfg.get("seed", 42))
    random_scores = {uid: random_trigger.score(uid) for uid in utt_ids}

    lambdas = [0.0, 0.1, 0.3, 1.0]
    triggers: dict[str, dict[str, float]] = {"random": random_scores}
    mi_results: dict[str, dict] = {}
    layer_weights_by_lambda: dict[str, list[float]] = {}

    for lam in lambdas:
        lam_str = f"{lam:.1f}".replace(".", "")
        model_path = models_dir / f"probe_multitask_lam{lam_str}.pt"

        if not model_path.exists():
            print(f"  Lambda={lam}: model not found at {model_path}")
            continue

        checkpoint = torch.load(model_path, weights_only=False)
        accent_to_idx = checkpoint.get("accent_to_idx", {})

        model = MultiTaskProbe(
            num_accents=len(accent_to_idx) if accent_to_idx else 6,
            num_layers=cfg["features"]["num_layers"],
            hidden_dim=cfg["features"]["hidden_dim"],
            probe_dim=checkpoint["config"]["hidden_dim"],
            dropout=checkpoint["config"]["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        calibration = checkpoint["calibration"]
        low, high = calibration["low"], calibration["high"]
        rng = high - low if high - low > 1e-8 else 1.0

        # Score using regression head
        scores: dict[str, float] = {}
        with torch.no_grad():
            for uid, feat in test_features.items():
                raw = model.predict_score(feat.unsqueeze(0)).item()
                normed = (raw - low) / rng
                scores[uid] = max(0.0, min(1.0, normed))

        trig_name = f"multitask_lam_{lam_str}"
        triggers[trig_name] = scores
        layer_weights_by_lambda[str(lam)] = checkpoint.get("layer_weights", [])

        # MI check
        leakage = speaker_leakage_check(
            list(scores.keys()), scores, speaker_map, accent_map
        )
        mi_results[str(lam)] = leakage
        print(f"  Lambda={lam}: MI(accent)={leakage['mi_score_accent']:.4f}, "
              f"MI(speaker)={leakage['mi_score_speaker']:.4f}")

    # Score all triggers
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

    # Paired bootstrap: best lambda vs champion single-task
    champion_path = Path(cfg["probe_gain"]["model_path"])
    if not champion_path.exists():
        champion_path = Path(cfg["probe"]["model_path"])

    if champion_path.exists() and test_features:
        checkpoint = torch.load(champion_path, weights_only=False)
        probe = AccentednessProbe(
            num_layers=cfg["features"]["num_layers"],
            hidden_dim=cfg["features"]["hidden_dim"],
            probe_dim=checkpoint["config"]["hidden_dim"],
            dropout=checkpoint["config"]["dropout"],
        )
        probe.load_state_dict(checkpoint["model_state_dict"])
        champion_trigger = ScalarProbeTrigger(probe, test_features, checkpoint["calibration"])
        champion_scores = {uid: champion_trigger.score(uid)
                           for uid in utt_ids if uid in test_features}

        # Find best lambda by area_vs_random
        best_lam_name = None
        best_avr = -float("inf")
        for trig_name, trig_data in results["triggers"].items():
            if trig_name.startswith("multitask_"):
                avr = trig_data["summary"].get("area_vs_random", -float("inf"))
                if avr is not None and avr > best_avr:
                    best_avr = avr
                    best_lam_name = trig_name

        if best_lam_name:
            print(f"\nPaired: {best_lam_name} vs champion")
            paired = paired_bootstrap(
                triggers[best_lam_name], champion_scores,
                default_wers, careful_wers,
                n=args.bootstrap_n, seed=cfg.get("seed", 42),
                random_scores=random_scores,
            )
            results["paired_best_vs_champion"] = {
                "best_lambda": best_lam_name,
                **paired,
            }

    results["mi_by_lambda"] = mi_results

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    setup_style()

    # MI vs lambda
    if mi_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        lam_vals = sorted(mi_results.keys(), key=float)
        mi_accent = [mi_results[l]["mi_score_accent"] for l in lam_vals]
        mi_speaker = [mi_results[l]["mi_score_speaker"] for l in lam_vals]

        ax.plot([float(l) for l in lam_vals], mi_accent, "o-", label="MI(score, accent)",
                color="blue", linewidth=2, markersize=8)
        ax.plot([float(l) for l in lam_vals], mi_speaker, "s--", label="MI(score, speaker)",
                color="red", linewidth=2, markersize=8)
        ax.set_xlabel("Lambda (accent loss weight)")
        ax.set_ylabel("Mutual Information")
        ax.set_title("MI vs Lambda: Does accent supervision dominate?")
        ax.legend()

        fig.tight_layout()
        fig.savefig(fig_dir / "mi_vs_lambda.png")
        plt.close(fig)
        print(f"Saved: {fig_dir / 'mi_vs_lambda.png'}")

    # Layer weights vs lambda
    if layer_weights_by_lambda:
        n_lambdas = len(layer_weights_by_lambda)
        fig, axes = plt.subplots(1, n_lambdas, figsize=(5 * n_lambdas, 4), sharey=True)
        if n_lambdas == 1:
            axes = [axes]

        for ax, (lam_str, weights) in zip(axes, sorted(layer_weights_by_lambda.items(),
                                                         key=lambda x: float(x[0]))):
            layers = list(range(len(weights)))
            ax.bar(layers, weights, color=sns.color_palette("viridis", len(weights)))
            ax.set_xlabel("WavLM Layer")
            ax.set_title(f"Lambda={lam_str}")
            if ax == axes[0]:
                ax.set_ylabel("Weight")

        fig.suptitle("Layer Weights by Lambda", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "layer_weights_by_lambda.png")
        plt.close(fig)
        print(f"Saved: {fig_dir / 'layer_weights_by_lambda.png'}")

    # Save
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
