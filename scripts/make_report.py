"""Generate a markdown report from experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-02-scalar-vs-baselines"
    metrics_path = exp_dir / "metrics.json"

    if not metrics_path.exists():
        print(f"No metrics found at {metrics_path}. Run `make eval` first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    report_lines = [
        "# Experiment Report: Scalar Probe vs Baselines",
        "",
        "## Operating Curve Summary",
        "",
        "| Trigger | WER@10% | WER@20% | WER@30% | WER@50% | Area vs Random |",
        "|---------|---------|---------|---------|---------|----------------|",
    ]

    for name, summary in metrics.get("summaries", {}).items():
        row = f"| {name} "
        for bp in [10, 20, 30, 50]:
            val = summary.get(f"net_wer_at_{bp}pct")
            row += f"| {val:.4f} " if val is not None else "| N/A "
        avr = summary.get("area_vs_random")
        row += f"| {avr:.4f} |" if avr is not None else "| N/A |"
        report_lines.append(row)

    # Per-accent breakdown
    per_accent = metrics.get("per_accent", {})
    if per_accent:
        report_lines.extend([
            "",
            "## Per-Accent WER",
            "",
            "| Accent | Default WER | Careful WER | Escalation Gain |",
            "|--------|-------------|-------------|-----------------|",
        ])
        for accent, data in sorted(per_accent.items()):
            report_lines.append(
                f"| {accent} | {data['mean_default_wer']:.4f} | "
                f"{data['mean_careful_wer']:.4f} | "
                f"{data['mean_escalation_gain']:.4f} |"
            )

    # Dominance check
    dominance = metrics.get("dominance", {})
    scalar_dom = dominance.get("scalar_probe_dominates_argmax_accent")
    if scalar_dom is not None:
        report_lines.extend([
            "",
            "## Key Result",
            "",
            f"Scalar probe {'**dominates**' if scalar_dom else 'does **not** dominate'} "
            f"the argmax accent baseline.",
        ])

    # Leakage
    leakage = metrics.get("leakage")
    if leakage:
        report_lines.extend([
            "",
            "## Speaker Leakage Check",
            "",
            f"- MI(score, speaker) = {leakage['mi_score_speaker']:.4f}",
            f"- MI(score, accent) = {leakage['mi_score_accent']:.4f}",
            f"- Ratio = {leakage['ratio_speaker_to_accent']:.2f}",
        ])

    report_lines.extend([
        "",
        "## Figures",
        "",
        "- [Operating Curves](figures/operating_curves.png)",
        "- [Layer Weights](figures/layer_weights.png)",
        "- [Score Distributions](figures/score_distributions.png)",
        "- [Per-Accent WER](figures/per_accent_wer.png)",
    ])

    report_path = exp_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
