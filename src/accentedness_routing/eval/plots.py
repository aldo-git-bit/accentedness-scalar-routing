"""Publication-quality plots for operating curves and analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def plot_operating_curves(
    curves: dict[str, dict],
    output_path: str,
    title: str = "Operating Curves: Net WER vs Escalation Rate",
):
    """Plot all operating curves on a single figure."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Style mapping
    styles = {
        "oracle": {"color": "green", "linestyle": "-", "linewidth": 2, "alpha": 0.8},
        "scalar_probe": {"color": "blue", "linestyle": "-", "linewidth": 2.5, "alpha": 1.0},
        "argmax_accent": {"color": "orange", "linestyle": "--", "linewidth": 2, "alpha": 0.8},
        "confidence": {"color": "purple", "linestyle": "-.", "linewidth": 1.5, "alpha": 0.7},
        "random": {"color": "gray", "linestyle": ":", "linewidth": 1.5, "alpha": 0.6},
    }

    for name, curve in curves.items():
        if name in ("default_only", "careful_only"):
            continue

        rates = curve["escalation_rates"]
        wers = curve["net_wers"]

        # Sort by escalation rate for clean plotting
        order = np.argsort(rates)
        rates = np.array(rates)[order]
        wers = np.array(wers)[order]

        style = styles.get(name, {"color": "black", "linestyle": "-", "linewidth": 1})
        ax.plot(rates, wers, label=name.replace("_", " ").title(), **style)

    # Add floor lines
    if "default_only" in curves:
        ax.axhline(
            y=curves["default_only"]["net_wers"][0],
            color="red", linestyle=":", alpha=0.5, label="Default only",
        )
    if "careful_only" in curves:
        ax.axhline(
            y=curves["careful_only"]["net_wers"][0],
            color="green", linestyle=":", alpha=0.5, label="Careful only",
        )

    ax.set_xlabel("Escalation Rate")
    ax.set_ylabel("Net WER")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_layer_weights(
    weights: list[float],
    output_path: str,
    title: str = "WavLM Layer Weights (Learned)",
):
    """Bar chart of learned layer weights."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    layers = list(range(len(weights)))
    colors = sns.color_palette("viridis", len(weights))
    ax.bar(layers, weights, color=colors)
    ax.set_xlabel("WavLM Layer")
    ax.set_ylabel("Weight (softmax normalized)")
    ax.set_title(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_score_distributions(
    scores_by_accent: dict[str, list[float]],
    output_path: str,
    title: str = "Scalar Score Distribution by Accent",
):
    """Histogram of scalar scores by accent."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for accent, scores in sorted(scores_by_accent.items()):
        ax.hist(scores, bins=20, alpha=0.5, label=accent, density=True)

    ax.set_xlabel("Scalar Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_accent_delta(
    per_accent: dict[str, dict],
    output_path: str,
    title: str = "Per-Accent WER: Default vs Careful",
):
    """Grouped bar chart of per-accent WER."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    accents = sorted(per_accent.keys())
    x = np.arange(len(accents))
    width = 0.35

    default_wers = [per_accent[a]["mean_default_wer"] for a in accents]
    careful_wers = [per_accent[a]["mean_careful_wer"] for a in accents]

    ax.bar(x - width / 2, default_wers, width, label="Default (Whisper-small)", color="salmon")
    ax.bar(x + width / 2, careful_wers, width, label="Careful (Whisper-large-v3)", color="skyblue")

    ax.set_xlabel("Accent")
    ax.set_ylabel("Mean WER")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace(" ", "\n") for a in accents], rotation=0, fontsize=9)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")
