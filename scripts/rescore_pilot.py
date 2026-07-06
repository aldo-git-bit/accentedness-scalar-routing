"""Re-score all 5 pilot triggers through eval_common with capped WER + CIs.

Produces: experiments/EXP-00-rescore/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

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
from accentedness_routing.triggers.oracle import OracleTrigger
from accentedness_routing.triggers.random_trigger import RandomTrigger
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger


def main():
    parser = argparse.ArgumentParser(description="Re-score pilot triggers with capped WER + CIs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-00-rescore"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load test utterances
    with open(data_dir / "test_utterances.pkl", "rb") as f:
        test_utts = pickle.load(f)
    print(f"Loaded {len(test_utts)} test utterances")

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
    print(f"  {len(utt_ids)} utterances with ASR results")

    # Compute capped escalation gains
    gains = {uid: escalation_gain(default_wers[uid], careful_wers[uid]) for uid in utt_ids}

    # Per-accent mean WER for argmax trigger
    accent_wer_sums: dict[str, float] = defaultdict(float)
    accent_counts: dict[str, int] = defaultdict(int)
    for uid in utt_ids:
        accent_wer_sums[accent_map[uid]] += default_wers[uid]
        accent_counts[accent_map[uid]] += 1
    accent_mean_wers = {a: accent_wer_sums[a] / accent_counts[a] for a in accent_wer_sums}

    # Build triggers
    oracle = OracleTrigger(default_wers, careful_wers)
    random_trigger = RandomTrigger(seed=cfg.get("seed", 42))
    confidence = ConfidenceTrigger(logprobs)
    argmax = ArgmaxAccentTrigger(accent_map, accent_mean_wers)

    triggers: dict[str, dict[str, float]] = {
        "oracle": {uid: oracle.score(uid) for uid in utt_ids},
        "random": {uid: random_trigger.score(uid) for uid in utt_ids},
        "argmax_accent": {uid: argmax.score(uid) for uid in utt_ids},
        "confidence": {uid: confidence.score(uid) for uid in utt_ids},
    }

    # Load scalar probe if available
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

        features_dir = Path(cfg["features"]["cache_dir"])
        test_features = {}
        for uid in utt_ids:
            fp = features_dir / f"{uid}.pt"
            if fp.exists():
                test_features[uid] = torch.load(fp, weights_only=True)

        if test_features:
            probe_trigger = ScalarProbeTrigger(probe, test_features, calibration)
            triggers["scalar_probe"] = {uid: probe_trigger.score(uid)
                                        for uid in utt_ids if uid in test_features}
            print(f"  Loaded scalar probe with {len(test_features)} test features")
    else:
        print("  No probe model found, skipping scalar_probe trigger")

    # Score all triggers through eval_common
    random_scores = triggers["random"]
    results: dict = {"triggers": {}}

    for trig_name, trig_scores in triggers.items():
        print(f"\nScoring: {trig_name} ({len(trig_scores)} utterances)")

        # Operating curve with capped WER
        curve = operating_curve(trig_scores, default_wers, careful_wers)
        random_curve = operating_curve(random_scores, default_wers, careful_wers)
        summary = summarize(curve, random_curve)

        # Decision scorecard at tau=0.0 and tau=0.05
        scorecard_0 = decision_scorecard(trig_scores, gains, tau=0.0)
        scorecard_05 = decision_scorecard(trig_scores, gains, tau=0.05)

        # Bootstrap CIs
        print(f"  Bootstrap ({args.bootstrap_n} resamples)...")
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
            "n_utterances": len(trig_scores),
        }

        # Print summary
        for k, v in summary.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")
        print(f"  AUC(tau=0): {scorecard_0.get('auc', 'N/A')}")
        print(f"  AUC(tau=0.05): {scorecard_05.get('auc', 'N/A')}")

    # Paired bootstrap: scalar_probe vs argmax_accent
    if "scalar_probe" in triggers and "argmax_accent" in triggers:
        print("\nPaired bootstrap: scalar_probe vs argmax_accent...")
        paired = paired_bootstrap(
            triggers["scalar_probe"], triggers["argmax_accent"],
            default_wers, careful_wers,
            n=args.bootstrap_n, seed=cfg.get("seed", 42),
            random_scores=random_scores,
        )
        results["paired_probe_vs_argmax"] = paired

    # Hallucination stats
    n_hallucinated = sum(1 for uid in utt_ids if default_wers[uid] > 1.0)
    n_careful_hallucinated = sum(1 for uid in utt_ids if careful_wers[uid] > 1.0)
    results["data_summary"] = {
        "n_test_utterances": len(utt_ids),
        "n_default_hallucinated": n_hallucinated,
        "n_careful_hallucinated": n_careful_hallucinated,
        "mean_default_wer_uncapped": float(sum(default_wers[uid] for uid in utt_ids) / len(utt_ids)),
        "mean_default_wer_capped": float(sum(cap_wer(default_wers[uid]) for uid in utt_ids) / len(utt_ids)),
        "mean_careful_wer_uncapped": float(sum(careful_wers[uid] for uid in utt_ids) / len(utt_ids)),
        "mean_careful_wer_capped": float(sum(cap_wer(careful_wers[uid]) for uid in utt_ids) / len(utt_ids)),
    }

    # Save
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
