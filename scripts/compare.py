"""Master comparison script.

Reads all experiments/EXP-*/metrics.json, produces:
  - experiments/COMPARISON.md (living leaderboard with CIs)
  - experiments/figures/operating_curves_all.png (master overlay with CI bands)
  - experiments/figures/headroom_sweep.png (headroom sweep across pairings, if grid data exists)

Supports both flat metrics.json (Rounds 1-2) and grid-schema metrics.json
with a top-level "pairing" key (Round 3 EXP-09+).
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


def load_experiment(exp_dir: Path) -> dict | None:
    """Load metrics.json from an experiment directory."""
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def load_grid_experiment(exp_dir: Path) -> list[tuple[str, dict]]:
    """Load per-pairing metrics.json from a grid experiment directory.

    Grid experiments have subdirectories per pairing, each with its own
    metrics.json containing a "pairing" key.

    Returns list of (pairing_slug, data) tuples.
    """
    results = []
    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            pairing = data.get("pairing", subdir.name)
            results.append((pairing, data))
    return results


def format_ci(point: float | None, ci_lo: float | None = None,
              ci_hi: float | None = None) -> str:
    """Format a value with optional CI."""
    if point is None:
        return "---"
    if ci_lo is not None and ci_hi is not None:
        return f"{point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"
    return f"{point:.4f}"


def collect_triggers(exp_dir: Path, data: dict, pairing: str | None = None) -> dict[str, dict]:
    """Extract trigger entries from a metrics.json, tagging with pairing if present."""
    triggers: dict[str, dict] = {}
    triggers_data = data.get("triggers", {})
    for trig_name, trig_data in triggers_data.items():
        suffix = f" [{pairing}]" if pairing else ""
        key = f"{trig_name} ({exp_dir.name}{suffix})"
        triggers[key] = {
            "experiment": exp_dir.name,
            "pairing": pairing,
            "trigger": trig_name,
            "summary": trig_data.get("summary", {}),
            "bootstrap_ci": trig_data.get("bootstrap_ci", {}),
            "curve_bands": trig_data.get("curve_bands"),
            "scorecard_tau_0.00": trig_data.get("scorecard_tau_0.00", {}),
            "scorecard_tau_0.05": trig_data.get("scorecard_tau_0.05", {}),
        }
    return triggers


def main():
    parser = argparse.ArgumentParser(description="Master comparison across experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_base = Path(cfg["output"]["experiments_dir"])
    fig_dir = exp_base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Discover all experiment directories
    exp_dirs = sorted(exp_base.glob("EXP-*"))
    print(f"Found {len(exp_dirs)} experiment directories")

    # Collect all triggers and their summaries across experiments
    all_triggers: dict[str, dict] = {}
    # Collect headroom data for sweep figure
    headroom_data: list[dict] = []

    for exp_dir in exp_dirs:
        # Try grid schema first (subdirectories with per-pairing metrics)
        grid_results = load_grid_experiment(exp_dir)
        if grid_results:
            print(f"  Loading {exp_dir.name} (grid: {len(grid_results)} pairings)")
            for pairing, data in grid_results:
                triggers = collect_triggers(exp_dir, data, pairing=pairing)
                all_triggers.update(triggers)

                # Collect headroom info if present
                headroom = data.get("headroom")
                if headroom:
                    # Find best learned trigger area_vs_random for this pairing
                    best_learned_avr = None
                    for trig_name, trig_data in data.get("triggers", {}).items():
                        if trig_name in ("oracle", "random", "confidence"):
                            continue
                        avr = trig_data.get("summary", {}).get("area_vs_random")
                        if avr is not None:
                            if best_learned_avr is None or avr > best_learned_avr:
                                best_learned_avr = avr

                    # Get confidence area_vs_random
                    conf_avr = None
                    conf_trig = data.get("triggers", {}).get("confidence", {})
                    if conf_trig:
                        conf_avr = conf_trig.get("summary", {}).get("area_vs_random")

                    headroom_data.append({
                        "pairing": pairing,
                        "wer_gap": headroom.get("wer_gap", 0),
                        "oracle_avr": headroom.get("oracle_area_vs_random", 0),
                        "confidence_avr": conf_avr,
                        "best_learned_avr": best_learned_avr,
                        "careful_model": pairing.split("__")[-1] if "__" in pairing else pairing,
                    })
            continue

        # Flat schema (single metrics.json)
        data = load_experiment(exp_dir)
        if data is None:
            print(f"  Skipping {exp_dir.name}: no metrics.json")
            continue

        print(f"  Loading {exp_dir.name}")
        pairing = data.get("pairing")
        triggers = collect_triggers(exp_dir, data, pairing=pairing)
        all_triggers.update(triggers)

    if not all_triggers:
        print("No experiment data found. Run experiments first.")
        return

    # -----------------------------------------------------------------------
    # Generate COMPARISON.md
    # -----------------------------------------------------------------------
    lines = [
        "# Experiment Comparison (Living Leaderboard)",
        "",
        f"*Auto-generated by `scripts/compare.py`. {len(all_triggers)} trigger entries.*",
        "",
    ]

    # Group by pairing if any triggers have pairing info
    has_pairings = any(t.get("pairing") for t in all_triggers.values())

    # Summary table
    if has_pairings:
        lines.extend([
            "## Operating Curve Summary",
            "",
            "| Trigger | Experiment | Pairing | netWER@10% | netWER@20% | netWER@30% | netWER@50% | Area vs Random |",
            "|---------|-----------|---------|------------|------------|------------|------------|---------------|",
        ])
    else:
        lines.extend([
            "## Operating Curve Summary",
            "",
            "| Trigger | Experiment | netWER@10% | netWER@20% | netWER@30% | netWER@50% | Area vs Random |",
            "|---------|-----------|------------|------------|------------|------------|---------------|",
        ])

    sorted_triggers = sorted(all_triggers.items(),
                             key=lambda x: (x[1].get("pairing") or "", x[1]["trigger"]))

    for key, tdata in sorted_triggers:
        summary = tdata["summary"]
        ci = tdata.get("bootstrap_ci", {})

        def _fmt(metric_key: str) -> str:
            point = summary.get(metric_key)
            ci_data = ci.get(metric_key, {})
            return format_ci(point, ci_data.get("ci_lo"), ci_data.get("ci_hi"))

        pairing_col = f"| {tdata.get('pairing') or '---'} " if has_pairings else ""
        row = (
            f"| {tdata['trigger']} | {tdata['experiment']} "
            f"{pairing_col}"
            f"| {_fmt('net_wer_at_10pct')} "
            f"| {_fmt('net_wer_at_20pct')} "
            f"| {_fmt('net_wer_at_30pct')} "
            f"| {_fmt('net_wer_at_50pct')} "
            f"| {_fmt('area_vs_random')} |"
        )
        lines.append(row)

    # Decision scorecard table
    if has_pairings:
        lines.extend([
            "",
            "## Decision Scorecard",
            "",
            "| Trigger | Experiment | Pairing | AUC (tau=0) | AP (tau=0) | AUC (tau=0.05) | AP (tau=0.05) | Pearson r | Spearman r |",
            "|---------|-----------|---------|-------------|-----------|----------------|--------------|-----------|-----------|",
        ])
    else:
        lines.extend([
            "",
            "## Decision Scorecard",
            "",
            "| Trigger | Experiment | AUC (tau=0) | AP (tau=0) | AUC (tau=0.05) | AP (tau=0.05) | Pearson r | Spearman r |",
            "|---------|-----------|-------------|-----------|----------------|--------------|-----------|-----------|",
        ])

    for key, tdata in sorted_triggers:
        sc0 = tdata.get("scorecard_tau_0.00", {})
        sc05 = tdata.get("scorecard_tau_0.05", {})

        def _fv(v):
            return f"{v:.4f}" if v is not None else "---"

        pairing_col = f"| {tdata.get('pairing') or '---'} " if has_pairings else ""
        row = (
            f"| {tdata['trigger']} | {tdata['experiment']} "
            f"{pairing_col}"
            f"| {_fv(sc0.get('auc'))} "
            f"| {_fv(sc0.get('ap'))} "
            f"| {_fv(sc05.get('auc'))} "
            f"| {_fv(sc05.get('ap'))} "
            f"| {_fv(sc0.get('pearson_r'))} "
            f"| {_fv(sc0.get('spearman_r'))} |"
        )
        lines.append(row)

    lines.append("")

    comparison_path = exp_base / "COMPARISON.md"
    with open(comparison_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {comparison_path}")

    # -----------------------------------------------------------------------
    # Master overlay figure with CI bands
    # -----------------------------------------------------------------------
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Style palette for triggers
    trigger_styles = {
        "oracle": {"color": "green", "linestyle": "-", "linewidth": 2, "alpha": 0.8},
        "scalar_probe": {"color": "blue", "linestyle": "-", "linewidth": 2.5, "alpha": 1.0},
        "argmax_accent": {"color": "orange", "linestyle": "--", "linewidth": 2, "alpha": 0.8},
        "confidence": {"color": "purple", "linestyle": "-.", "linewidth": 1.5, "alpha": 0.7},
        "random": {"color": "gray", "linestyle": ":", "linewidth": 1.5, "alpha": 0.6},
        "probe_gain": {"color": "red", "linestyle": "-", "linewidth": 2.5, "alpha": 1.0},
        "probe_capped_wer": {"color": "cyan", "linestyle": "--", "linewidth": 2, "alpha": 0.8},
        "compression_ratio": {"color": "brown", "linestyle": "-.", "linewidth": 1.5, "alpha": 0.7},
        "no_speech_prob": {"color": "pink", "linestyle": ":", "linewidth": 1.5, "alpha": 0.7},
        "probe_mean_std": {"color": "darkred", "linestyle": "-", "linewidth": 2.5, "alpha": 1.0},
        "probe_mean_only": {"color": "darkblue", "linestyle": "--", "linewidth": 2, "alpha": 0.8},
        "probe_std_only": {"color": "darkgreen", "linestyle": "-.", "linewidth": 1.5, "alpha": 0.7},
        "champion_retrained": {"color": "crimson", "linestyle": "-", "linewidth": 2, "alpha": 0.9},
        "combiner": {"color": "darkviolet", "linestyle": "-", "linewidth": 2, "alpha": 0.9},
    }

    plotted = set()
    for key, tdata in sorted(all_triggers.items()):
        trig_name = tdata["trigger"]
        bands = tdata.get("curve_bands")
        if bands is None:
            continue

        # Avoid plotting the same trigger from the same experiment twice
        plot_key = f"{trig_name}_{tdata['experiment']}_{tdata.get('pairing', '')}"
        if plot_key in plotted:
            continue
        plotted.add(plot_key)

        rates = np.array(bands["escalation_rates"])
        wers_point = np.array(bands["net_wers_point"])
        order = np.argsort(rates)

        style = trigger_styles.get(trig_name,
                                   {"color": "black", "linestyle": "-", "linewidth": 1})
        pairing_label = f" [{tdata['pairing']}]" if tdata.get("pairing") else ""
        label = f"{trig_name} ({tdata['experiment']}{pairing_label})"
        ax.plot(rates[order], wers_point[order], label=label, **style)

        # CI bands
        if "net_wers_ci_lo" in bands and "net_wers_ci_hi" in bands:
            ci_lo = np.array(bands["net_wers_ci_lo"])
            ci_hi = np.array(bands["net_wers_ci_hi"])
            ax.fill_between(rates[order], ci_lo[order], ci_hi[order],
                            alpha=0.1, color=style.get("color", "black"))

    ax.set_xlabel("Escalation Rate")
    ax.set_ylabel("Net WER (capped)")
    ax.set_title("Master Operating Curve Comparison (with 95% CI bands)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(fig_dir / "operating_curves_all.png")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'operating_curves_all.png'}")

    # -----------------------------------------------------------------------
    # Headroom sweep figure (Round 3)
    # -----------------------------------------------------------------------
    if headroom_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Separate by careful model for marker/color encoding
        careful_models = sorted(set(d["careful_model"] for d in headroom_data))
        markers = {"whisper_large_v3_mlx": "o", "whisper_large_v3_turbo": "s"}
        colors_map = {"whisper_large_v3_mlx": "blue", "whisper_large_v3_turbo": "orange"}

        for cm in careful_models:
            subset = [d for d in headroom_data if d["careful_model"] == cm]
            gaps = [d["wer_gap"] for d in subset]
            marker = markers.get(cm, "^")
            color = colors_map.get(cm, "gray")

            # Oracle
            oracle_avrs = [d["oracle_avr"] for d in subset]
            ax.scatter(gaps, oracle_avrs, marker=marker, color="green", s=80, zorder=3)
            ax.plot(gaps, oracle_avrs, color="green", alpha=0.5, linestyle="--")

            # Confidence
            conf_avrs = [d["confidence_avr"] if d["confidence_avr"] is not None else 0
                         for d in subset]
            ax.scatter(gaps, conf_avrs, marker=marker, color="purple", s=80, zorder=3)
            ax.plot(gaps, conf_avrs, color="purple", alpha=0.5, linestyle="--")

            # Best learned
            learned_avrs = [d["best_learned_avr"] if d["best_learned_avr"] is not None else 0
                            for d in subset]
            ax.scatter(gaps, learned_avrs, marker=marker, color=color, s=80, zorder=3,
                       label=f"best learned ({cm})")
            ax.plot(gaps, learned_avrs, color=color, alpha=0.5, linestyle="--")

        # Legend entries for oracle/confidence (just once)
        ax.scatter([], [], marker="o", color="green", s=80, label="oracle")
        ax.scatter([], [], marker="o", color="purple", s=80, label="confidence")

        ax.set_xlabel("WER Gap (mean capped default - mean capped careful)")
        ax.set_ylabel("Area vs Random")
        ax.set_title("Headroom Sweep: Does Wider Gap Create Exploitable Headroom?")
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(fig_dir / "headroom_sweep.png")
        plt.close(fig)
        print(f"Saved: {fig_dir / 'headroom_sweep.png'}")


if __name__ == "__main__":
    main()
