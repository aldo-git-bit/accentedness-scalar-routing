"""Extension 2 evaluation: Learning curve figure + layer weight comparison.

Produces: experiments/EXP-05-extension2-diagnostics/
  - learning_curve.png
  - layer_weights_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def main():
    parser = argparse.ArgumentParser(description="Extension 2: Evaluation figures")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-05-extension2-diagnostics"
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    # -----------------------------------------------------------------------
    # 1. Learning curve figure
    # -----------------------------------------------------------------------
    lc_path = exp_dir / "learning_curve.json"
    if lc_path.exists():
        with open(lc_path) as f:
            lc_data = json.load(f)

        fractions = [entry["fraction"] for entry in lc_data["fractions"]]
        n_utts = [entry["n_utterances"] for entry in lc_data["fractions"]]
        val_aucs = [entry["val_auc"] for entry in lc_data["fractions"]]
        val_pearson = [entry["val_pearson_r"] for entry in lc_data["fractions"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # AUC
        ax1.plot(fractions, val_aucs, "o-", color="blue", linewidth=2, markersize=8)
        for i, (frac, auc, n) in enumerate(zip(fractions, val_aucs, n_utts)):
            if auc is not None:
                ax1.annotate(f"n={n}", (frac, auc), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=8)
        ax1.set_xlabel("Fraction of Train Speakers")
        ax1.set_ylabel("Val AUC (gain > 0)")
        ax1.set_title("Learning Curve: AUC")
        ax1.set_xticks(fractions)
        ax1.set_xticklabels([f"{f:.0%}" for f in fractions])

        # Pearson r
        ax2.plot(fractions, val_pearson, "o-", color="red", linewidth=2, markersize=8)
        for i, (frac, r, n) in enumerate(zip(fractions, val_pearson, n_utts)):
            if r is not None:
                ax2.annotate(f"n={n}", (frac, r), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=8)
        ax2.set_xlabel("Fraction of Train Speakers")
        ax2.set_ylabel("Val Pearson r")
        ax2.set_title("Learning Curve: Pearson r")
        ax2.set_xticks(fractions)
        ax2.set_xticklabels([f"{f:.0%}" for f in fractions])

        fig.suptitle(f"Learning Curve (target={lc_data.get('target_type', 'unknown')})",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "learning_curve.png")
        plt.close(fig)
        print(f"Saved: {fig_dir / 'learning_curve.png'}")
    else:
        print(f"Learning curve data not found at {lc_path}")

    # -----------------------------------------------------------------------
    # 2. Layer weight comparison: gain probe vs accent classifier
    # -----------------------------------------------------------------------
    gain_weights = None
    accent_weights = None

    gain_model_path = Path(cfg["probe_gain"]["model_path"])
    if gain_model_path.exists():
        checkpoint = torch.load(gain_model_path, weights_only=False)
        gain_weights = checkpoint.get("layer_weights")

    accent_path = exp_dir / "accent_classifier.json"
    if accent_path.exists():
        with open(accent_path) as f:
            accent_data = json.load(f)
        accent_weights = accent_data.get("layer_weights")

    if gain_weights and accent_weights:
        fig, ax = plt.subplots(figsize=(12, 5))
        layers = np.arange(len(gain_weights))
        width = 0.35

        ax.bar(layers - width / 2, gain_weights, width, label="Gain Probe",
               color="blue", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.bar(layers + width / 2, accent_weights, width, label="Accent Classifier",
               color="orange", alpha=0.7, edgecolor="black", linewidth=0.5)

        ax.set_xlabel("WavLM Layer")
        ax.set_ylabel("Weight (softmax normalized)")
        ax.set_title("Layer Specialization: Gain Probe vs Accent Classifier")
        ax.legend()
        ax.set_xticks(layers)

        fig.tight_layout()
        fig.savefig(fig_dir / "layer_weights_comparison.png")
        plt.close(fig)
        print(f"Saved: {fig_dir / 'layer_weights_comparison.png'}")
    else:
        print("One or both models not available for layer weight comparison")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary = {}
    if lc_path.exists():
        with open(lc_path) as f:
            lc_data = json.load(f)
        last = lc_data["fractions"][-1]
        second_last = lc_data["fractions"][-2] if len(lc_data["fractions"]) > 1 else None

        summary["learning_curve"] = {
            "final_auc": last.get("val_auc"),
            "final_pearson_r": last.get("val_pearson_r"),
        }
        if second_last and last.get("val_auc") and second_last.get("val_auc"):
            slope = last["val_auc"] - second_last["val_auc"]
            summary["learning_curve"]["auc_slope_75_100"] = slope
            summary["learning_curve"]["still_climbing"] = slope > 0.005

    if gain_weights and accent_weights:
        # Compute correlation between weight vectors
        from scipy.stats import pearsonr
        r, p = pearsonr(gain_weights, accent_weights)
        summary["layer_weight_correlation"] = {
            "pearson_r": float(r),
            "p_value": float(p),
            "gain_top3_layers": sorted(range(len(gain_weights)),
                                       key=lambda i: -gain_weights[i])[:3],
            "accent_top3_layers": sorted(range(len(accent_weights)),
                                         key=lambda i: -accent_weights[i])[:3],
        }

    with open(exp_dir / "ext2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {exp_dir / 'ext2_summary.json'}")


if __name__ == "__main__":
    main()
