"""Diagnostic analyses from cached data.

Produces: experiments/EXP-00-rescore/diagnostics/
  - wer_distribution.png: uncapped vs capped WER histogram
  - hallucination_by_accent.png: count of WER > 1.0 per accent
  - per_accent_gain.png: capped escalation gain distribution
  - confidence_vs_wer.png: avg_logprob vs cap_wer colored by accent
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import cap_wer, escalation_gain


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def main():
    parser = argparse.ArgumentParser(description="Diagnostic analyses from cached data")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    diag_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-00-rescore" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Load test utterances
    with open(data_dir / "test_utterances.pkl", "rb") as f:
        test_utts = pickle.load(f)

    # Load ASR results
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    default_wers: dict[str, float] = {}
    careful_wers: dict[str, float] = {}
    logprobs: dict[str, float] = {}
    accent_map: dict[str, str] = {}

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

    utt_ids = sorted(default_wers.keys())
    print(f"Loaded {len(utt_ids)} test utterances with ASR results")

    setup_style()

    # -----------------------------------------------------------------------
    # 1. Histogram: uncapped vs capped WER distribution
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    uncapped = [default_wers[uid] for uid in utt_ids]
    capped = [cap_wer(default_wers[uid]) for uid in utt_ids]

    axes[0].hist(uncapped, bins=50, color="salmon", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0].axvline(x=1.0, color="red", linestyle="--", label="WER = 1.0")
    axes[0].set_xlabel("WER (uncapped)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Default Model WER Distribution (Uncapped)")
    axes[0].legend()

    axes[1].hist(capped, bins=50, color="skyblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("WER (capped at 1.0)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Default Model WER Distribution (Capped)")

    fig.tight_layout()
    fig.savefig(diag_dir / "wer_distribution.png")
    plt.close(fig)
    print(f"Saved: {diag_dir / 'wer_distribution.png'}")

    # -----------------------------------------------------------------------
    # 2. Hallucination prevalence by accent
    # -----------------------------------------------------------------------
    accent_groups: dict[str, list[str]] = defaultdict(list)
    for uid in utt_ids:
        accent_groups[accent_map[uid]].append(uid)

    accents_sorted = sorted(accent_groups.keys())
    hall_counts = []
    total_counts = []
    for accent in accents_sorted:
        uids = accent_groups[accent]
        n_hall = sum(1 for uid in uids if default_wers[uid] > 1.0)
        hall_counts.append(n_hall)
        total_counts.append(len(uids))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(accents_sorted))
    bars = ax.bar(x, hall_counts, color="coral", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add total count labels
    for i, (h, t) in enumerate(zip(hall_counts, total_counts)):
        ax.text(i, h + 0.3, f"{h}/{t}\n({100*h/t:.0f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Accent")
    ax.set_ylabel("Utterances with WER > 1.0")
    ax.set_title("Hallucination Prevalence by Accent (Default Model)")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace(" ", "\n") for a in accents_sorted], fontsize=9)

    fig.tight_layout()
    fig.savefig(diag_dir / "hallucination_by_accent.png")
    plt.close(fig)
    print(f"Saved: {diag_dir / 'hallucination_by_accent.png'}")

    # -----------------------------------------------------------------------
    # 3. Per-accent escalation gain distribution (capped)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    gain_data = []
    gain_labels = []
    for accent in accents_sorted:
        uids = accent_groups[accent]
        accent_gains = [escalation_gain(default_wers[uid], careful_wers[uid]) for uid in uids]
        gain_data.append(accent_gains)
        gain_labels.append(accent)

    bp = ax.boxplot(gain_data, tick_labels=[a.replace(" ", "\n") for a in gain_labels],
                    patch_artist=True, showfliers=True)

    colors = sns.color_palette("Set2", len(accents_sorted))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Accent")
    ax.set_ylabel("Escalation Gain (capped)")
    ax.set_title("Per-Accent Escalation Gain Distribution")

    fig.tight_layout()
    fig.savefig(diag_dir / "per_accent_gain.png")
    plt.close(fig)
    print(f"Saved: {diag_dir / 'per_accent_gain.png'}")

    # -----------------------------------------------------------------------
    # 4. Scatter: avg_logprob vs cap_wer(default_wer), colored by accent
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    accent_colors = dict(zip(accents_sorted, sns.color_palette("tab10", len(accents_sorted))))
    for accent in accents_sorted:
        uids = accent_groups[accent]
        x_vals = [logprobs[uid] for uid in uids]
        y_vals = [cap_wer(default_wers[uid]) for uid in uids]
        ax.scatter(x_vals, y_vals, alpha=0.5, s=20, label=accent, color=accent_colors[accent])

    ax.set_xlabel("avg_logprob (default model)")
    ax.set_ylabel("WER (capped at 1.0)")
    ax.set_title("Confidence vs Capped WER by Accent\n(Preview: Ext 3 confidence autopsy)")
    ax.legend(fontsize=8, markerscale=2)

    fig.tight_layout()
    fig.savefig(diag_dir / "confidence_vs_wer.png")
    plt.close(fig)
    print(f"Saved: {diag_dir / 'confidence_vs_wer.png'}")

    # -----------------------------------------------------------------------
    # Summary stats JSON
    # -----------------------------------------------------------------------
    stats: dict = {
        "n_test_utterances": len(utt_ids),
        "per_accent": {},
    }
    for accent in accents_sorted:
        uids = accent_groups[accent]
        d_wers = [default_wers[uid] for uid in uids]
        c_wers = [careful_wers[uid] for uid in uids]
        gains = [escalation_gain(default_wers[uid], careful_wers[uid]) for uid in uids]
        lps = [logprobs[uid] for uid in uids]
        n_hall = sum(1 for uid in uids if default_wers[uid] > 1.0)

        stats["per_accent"][accent] = {
            "n_utterances": len(uids),
            "n_hallucinated": n_hall,
            "mean_default_wer_uncapped": float(np.mean(d_wers)),
            "mean_default_wer_capped": float(np.mean([cap_wer(w) for w in d_wers])),
            "mean_careful_wer": float(np.mean(c_wers)),
            "mean_escalation_gain": float(np.mean(gains)),
            "median_escalation_gain": float(np.median(gains)),
            "mean_avg_logprob": float(np.mean(lps)),
        }

    with open(diag_dir / "diagnostics_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDiagnostics saved to {diag_dir}")


if __name__ == "__main__":
    main()
