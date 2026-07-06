"""Extension 3: Hallucination baselines + confidence autopsy.

Evaluates compression_ratio and no_speech_prob triggers, plus a union trigger.
Includes confidence autopsy: do hallucinated utterances have higher avg_logprob?

Produces: experiments/EXP-06-extension3-hallucination/
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
import torch
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import (
    bootstrap,
    cap_wer,
    decision_scorecard,
    escalation_gain,
    operating_curve,
    paired_bootstrap,
    summarize,
)
from accentedness_routing.triggers.commonaccent import ArgmaxAccentTrigger
from accentedness_routing.triggers.confidence import ConfidenceTrigger
from accentedness_routing.triggers.hallucination import CompressionRatioTrigger, NoSpeechProbTrigger
from accentedness_routing.triggers.oracle import OracleTrigger
from accentedness_routing.triggers.random_trigger import RandomTrigger
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def main():
    parser = argparse.ArgumentParser(
        description="Extension 3: Hallucination baselines + confidence autopsy")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-06-extension3-hallucination"
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
    logprobs: dict[str, float] = {}
    no_speech_probs: dict[str, float] = {}
    hypotheses: dict[str, str] = {}
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
        no_speech_probs[uid] = d.get("no_speech_prob", 0.0)
        hypotheses[uid] = d.get("text", "")
        accent_map[uid] = utt.accent

    utt_ids = sorted(default_wers.keys())
    print(f"Loaded {len(utt_ids)} test utterances")

    gains = {uid: escalation_gain(default_wers[uid], careful_wers[uid]) for uid in utt_ids}

    # Build triggers
    compression_trigger = CompressionRatioTrigger(hypotheses)
    nospeech_trigger = NoSpeechProbTrigger(no_speech_probs)
    random_trigger = RandomTrigger(seed=cfg.get("seed", 42))

    triggers: dict[str, dict[str, float]] = {
        "compression_ratio": {uid: compression_trigger.score(uid) for uid in utt_ids},
        "no_speech_prob": {uid: nospeech_trigger.score(uid) for uid in utt_ids},
        "random": {uid: random_trigger.score(uid) for uid in utt_ids},
        "oracle": {uid: OracleTrigger(default_wers, careful_wers).score(uid) for uid in utt_ids},
        "confidence": {uid: ConfidenceTrigger(logprobs).score(uid) for uid in utt_ids},
    }

    # Union trigger: max(compression, no_speech_prob)
    triggers["hallucination_union"] = {
        uid: max(triggers["compression_ratio"][uid], triggers["no_speech_prob"][uid])
        for uid in utt_ids
    }

    # Load champion scalar probe if available
    champion_path = Path(cfg["probe_gain"]["model_path"])
    if not champion_path.exists():
        champion_path = Path(cfg["probe"]["model_path"])

    if champion_path.exists():
        features_dir = Path(cfg["features"]["cache_dir"])
        test_features = {}
        for uid in utt_ids:
            fp = features_dir / f"{uid}.pt"
            if fp.exists():
                test_features[uid] = torch.load(fp, weights_only=True)

        if test_features:
            checkpoint = torch.load(champion_path, weights_only=False)
            probe = AccentednessProbe(
                num_layers=cfg["features"]["num_layers"],
                hidden_dim=cfg["features"]["hidden_dim"],
                probe_dim=checkpoint["config"]["hidden_dim"],
                dropout=checkpoint["config"]["dropout"],
            )
            probe.load_state_dict(checkpoint["model_state_dict"])
            probe_trigger = ScalarProbeTrigger(probe, test_features, checkpoint["calibration"])
            champion_name = "champion_scalar"
            triggers[champion_name] = {uid: probe_trigger.score(uid) for uid in utt_ids
                                       if uid in test_features}

    # Score all triggers
    random_scores = triggers["random"]
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

    # Paired bootstrap: hallucination triggers vs champion
    if "champion_scalar" in triggers:
        for hall_name in ["compression_ratio", "no_speech_prob", "hallucination_union"]:
            print(f"\nPaired: {hall_name} vs champion_scalar")
            paired = paired_bootstrap(
                triggers[hall_name], triggers["champion_scalar"],
                default_wers, careful_wers,
                n=args.bootstrap_n, seed=cfg.get("seed", 42),
                random_scores=random_scores,
            )
            results[f"paired_{hall_name}_vs_champion"] = paired

    # -----------------------------------------------------------------------
    # Confidence autopsy
    # -----------------------------------------------------------------------
    setup_style()

    hallucinated_uids = [uid for uid in utt_ids if default_wers[uid] > 1.0]
    non_hallucinated_uids = [uid for uid in utt_ids if default_wers[uid] <= 1.0]

    hall_logprobs = [logprobs[uid] for uid in hallucinated_uids]
    non_hall_logprobs = [logprobs[uid] for uid in non_hallucinated_uids]
    median_logprob = float(np.median([logprobs[uid] for uid in utt_ids]))

    autopsy: dict = {
        "n_hallucinated": len(hallucinated_uids),
        "n_non_hallucinated": len(non_hallucinated_uids),
        "median_logprob_all": median_logprob,
    }

    if hall_logprobs:
        autopsy["mean_logprob_hallucinated"] = float(np.mean(hall_logprobs))
        autopsy["mean_logprob_non_hallucinated"] = float(np.mean(non_hall_logprobs))
        # Do hallucinated utterances have HIGHER avg_logprob than median?
        n_above_median = sum(1 for lp in hall_logprobs if lp > median_logprob)
        autopsy["hallucinated_above_median_pct"] = n_above_median / len(hall_logprobs) * 100
        autopsy["whisper_hallucinates_confidently"] = autopsy["hallucinated_above_median_pct"] > 50

    results["confidence_autopsy"] = autopsy

    # Scatter: avg_logprob vs cap_wer, colored by accent
    fig, ax = plt.subplots(figsize=(10, 7))
    accent_groups: dict[str, list[str]] = defaultdict(list)
    for uid in utt_ids:
        accent_groups[accent_map[uid]].append(uid)

    accents_sorted = sorted(accent_groups.keys())
    colors = dict(zip(accents_sorted, sns.color_palette("tab10", len(accents_sorted))))

    for accent in accents_sorted:
        uids = accent_groups[accent]
        x_vals = [logprobs[uid] for uid in uids]
        y_vals = [cap_wer(default_wers[uid]) for uid in uids]
        ax.scatter(x_vals, y_vals, alpha=0.5, s=20, label=accent, color=colors[accent])

    # Highlight hallucinated
    if hallucinated_uids:
        hall_x = [logprobs[uid] for uid in hallucinated_uids]
        hall_y = [1.0] * len(hallucinated_uids)  # all capped at 1.0
        ax.scatter(hall_x, hall_y, marker="x", s=60, color="red", linewidths=2,
                   label="Hallucinated (WER>1)", zorder=5)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.3)
    ax.axvline(x=median_logprob, color="gray", linestyle=":", alpha=0.5, label="Median logprob")
    ax.set_xlabel("avg_logprob (default model)")
    ax.set_ylabel("WER (capped at 1.0)")
    ax.set_title("Confidence Autopsy: avg_logprob vs Capped WER")
    ax.legend(fontsize=7, markerscale=2)

    fig.tight_layout()
    fig.savefig(fig_dir / "confidence_autopsy.png")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'confidence_autopsy.png'}")

    # Indian English slice analysis
    indian_uids = [uid for uid in utt_ids if "indian" in accent_map[uid].lower()
                   or "Indian" in accent_map[uid]]
    us_uids = [uid for uid in utt_ids if "american" in accent_map[uid].lower()
               or "US" in accent_map[uid] or "Mainstream" in accent_map[uid]]

    if indian_uids and us_uids:
        indian_hall = [uid for uid in indian_uids if default_wers[uid] > 1.0]
        us_hall = [uid for uid in us_uids if default_wers[uid] > 1.0]

        results["indian_english_slice"] = {
            "n_indian": len(indian_uids),
            "n_indian_hallucinated": len(indian_hall),
            "n_us": len(us_uids),
            "n_us_hallucinated": len(us_hall),
            "indian_mean_gain_capped": float(np.mean([gains[uid] for uid in indian_uids])),
            "us_mean_gain_capped": float(np.mean([gains[uid] for uid in us_uids])),
        }

    # Save
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
