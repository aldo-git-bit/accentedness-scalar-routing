"""Produce the headroom sweep figure from EXP-09 results.

experiments/figures/headroom_sweep.png:
  x-axis = WER gap (or oracle area) per pairing
  y-axis = area-vs-random
  Series: oracle, confidence, best-learned-trigger (champion retrained)
  Marker/color encodes careful model (turbo vs large-v3)
  Error bars from bootstrap CIs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def main():
    parser = argparse.ArgumentParser(description="Headroom sweep figure")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-09-headroom-grid"
    fig_dir = Path(cfg["output"]["experiments_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        print(f"No EXP-09 results found at {exp_dir}. Run eval_headroom_grid.py first.")
        return

    # Collect data from each pairing subdirectory
    pairing_data = []
    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        headroom = data.get("headroom", {})
        triggers = data.get("triggers", {})
        pairing = data.get("pairing", subdir.name)

        # Parse careful model from pairing slug
        parts = pairing.split("__")
        careful_model = parts[-1] if len(parts) > 1 else pairing
        default_model = parts[0] if len(parts) > 1 else "unknown"

        # Get area_vs_random for oracle, confidence, and best learned trigger
        oracle_avr = triggers.get("oracle", {}).get("summary", {}).get("area_vs_random")
        conf_avr = triggers.get("confidence", {}).get("summary", {}).get("area_vs_random")

        # Bootstrap CIs
        oracle_ci = triggers.get("oracle", {}).get("bootstrap_ci", {}).get(
            "area_vs_random", {})
        conf_ci = triggers.get("confidence", {}).get("bootstrap_ci", {}).get(
            "area_vs_random", {})

        # Best learned trigger (not oracle, random, or confidence)
        best_learned_avr = None
        best_learned_ci = {}
        best_learned_name = None
        for trig_name, trig_data in triggers.items():
            if trig_name in ("oracle", "random", "confidence"):
                continue
            avr = trig_data.get("summary", {}).get("area_vs_random")
            if avr is not None:
                if best_learned_avr is None or avr > best_learned_avr:
                    best_learned_avr = avr
                    best_learned_ci = trig_data.get("bootstrap_ci", {}).get(
                        "area_vs_random", {})
                    best_learned_name = trig_name

        pairing_data.append({
            "pairing": pairing,
            "default_model": default_model,
            "careful_model": careful_model,
            "wer_gap": headroom.get("wer_gap", 0),
            "oracle_avr": oracle_avr,
            "oracle_ci_lo": oracle_ci.get("ci_lo"),
            "oracle_ci_hi": oracle_ci.get("ci_hi"),
            "conf_avr": conf_avr,
            "conf_ci_lo": conf_ci.get("ci_lo"),
            "conf_ci_hi": conf_ci.get("ci_hi"),
            "best_learned_avr": best_learned_avr,
            "best_learned_ci_lo": best_learned_ci.get("ci_lo"),
            "best_learned_ci_hi": best_learned_ci.get("ci_hi"),
            "best_learned_name": best_learned_name,
        })

    if not pairing_data:
        print("No pairing data found.")
        return

    print(f"Loaded {len(pairing_data)} pairings for headroom sweep")

    # Sort by WER gap
    pairing_data.sort(key=lambda d: d["wer_gap"])

    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot by careful model type for marker differentiation
    careful_types = sorted(set(d["careful_model"] for d in pairing_data))
    markers = {"whisper-large-v3-mlx": "o", "whisper-large-v3-turbo": "s"}

    for cm in careful_types:
        subset = [d for d in pairing_data if d["careful_model"] == cm]
        gaps = [d["wer_gap"] for d in subset]
        marker = markers.get(cm, "^")

        # Oracle
        oracle_avrs = [d["oracle_avr"] or 0 for d in subset]
        oracle_errs = np.array([
            [max(0, (d["oracle_avr"] or 0) - (d["oracle_ci_lo"] or 0)) for d in subset],
            [max(0, (d["oracle_ci_hi"] or 0) - (d["oracle_avr"] or 0)) for d in subset],
        ])
        ax.errorbar(gaps, oracle_avrs, yerr=oracle_errs,
                     marker=marker, color="green", capsize=4, linewidth=1.5,
                     linestyle="--", alpha=0.8,
                     label=f"oracle ({cm})" if cm == careful_types[0] else "")

        # Confidence
        conf_avrs = [d["conf_avr"] or 0 for d in subset]
        conf_errs = np.array([
            [max(0, (d["conf_avr"] or 0) - (d["conf_ci_lo"] or 0)) for d in subset],
            [max(0, (d["conf_ci_hi"] or 0) - (d["conf_avr"] or 0)) for d in subset],
        ])
        ax.errorbar(gaps, conf_avrs, yerr=conf_errs,
                     marker=marker, color="purple", capsize=4, linewidth=1.5,
                     linestyle="-.", alpha=0.8,
                     label=f"confidence ({cm})" if cm == careful_types[0] else "")

        # Best learned
        learned_avrs = [d["best_learned_avr"] or 0 for d in subset]
        learned_errs = np.array([
            [max(0, (d["best_learned_avr"] or 0) - (d["best_learned_ci_lo"] or 0))
             for d in subset],
            [max(0, (d["best_learned_ci_hi"] or 0) - (d["best_learned_avr"] or 0))
             for d in subset],
        ])
        color = "blue" if "large-v3-mlx" in cm else "orange"
        ax.errorbar(gaps, learned_avrs, yerr=learned_errs,
                     marker=marker, color=color, capsize=4, linewidth=1.5,
                     linestyle="-", alpha=0.9,
                     label=f"best learned ({cm})")

    # Add legend entries for oracle/confidence
    ax.plot([], [], color="green", linestyle="--", linewidth=1.5, label="oracle")
    ax.plot([], [], color="purple", linestyle="-.", linewidth=1.5, label="confidence")

    # Add pairing labels
    for d in pairing_data:
        ax.annotate(d["default_model"][:4],
                    (d["wer_gap"], d["oracle_avr"] or 0),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.6)

    ax.set_xlabel("WER Gap (mean capped default - mean capped careful)")
    ax.set_ylabel("Area vs Random")
    ax.set_title("Headroom Sweep: Does Wider Gap Create Exploitable Headroom?")
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)

    fig.tight_layout()
    out_path = fig_dir / "headroom_sweep.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
