"""Extension 1 evaluation: gain-target probe vs pilot vs argmax.

Scores gain_probe, capped_wer_probe, pilot_probe, and all reference triggers
through eval_common with bootstrap CIs and paired bootstrap tests.

Produces: experiments/EXP-04-extension1-gain-target/
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


def load_probe_trigger(model_path: Path, cfg: dict, test_features: dict,
                       name_override: str | None = None) -> ScalarProbeTrigger | None:
    """Load a probe checkpoint and create a ScalarProbeTrigger."""
    if not model_path.exists():
        print(f"  Probe not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, weights_only=False)
    probe = AccentednessProbe(
        num_layers=cfg["features"]["num_layers"],
        hidden_dim=cfg["features"]["hidden_dim"],
        probe_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    )
    probe.load_state_dict(checkpoint["model_state_dict"])
    calibration = checkpoint["calibration"]

    trigger = ScalarProbeTrigger(probe, test_features, calibration)

    # Override the name if needed
    if name_override:
        trigger._name_override = name_override
        # Monkey-patch the name property
        trigger.__class__ = type(
            f"Named_{name_override}",
            (ScalarProbeTrigger,),
            {"name": property(lambda self: self._name_override)},
        )

    return trigger


def main():
    parser = argparse.ArgumentParser(description="Extension 1: Evaluate gain-target probe")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-04-extension1-gain-target"
    exp_dir.mkdir(parents=True, exist_ok=True)

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

    # Capped escalation gains
    gains = {uid: escalation_gain(default_wers[uid], careful_wers[uid]) for uid in utt_ids}

    # Load test features
    features_dir = Path(cfg["features"]["cache_dir"])
    test_features = {}
    for uid in utt_ids:
        fp = features_dir / f"{uid}.pt"
        if fp.exists():
            test_features[uid] = torch.load(fp, weights_only=True)
    print(f"  {len(test_features)} test features loaded")

    # Per-accent mean WER for argmax
    accent_wer_sums: dict[str, float] = defaultdict(float)
    accent_counts: dict[str, int] = defaultdict(int)
    for uid in utt_ids:
        accent_wer_sums[accent_map[uid]] += default_wers[uid]
        accent_counts[accent_map[uid]] += 1
    accent_mean_wers = {a: accent_wer_sums[a] / accent_counts[a] for a in accent_wer_sums}

    # Build all triggers
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

    # Load probe triggers
    probe_configs = [
        ("probe_gain", Path(cfg["probe_gain"]["model_path"])),
        ("probe_capped_wer", Path(cfg["probe_capped_wer"]["model_path"])),
        ("scalar_probe", Path(cfg["probe"]["model_path"])),  # pilot
    ]

    for name, mpath in probe_configs:
        trigger = load_probe_trigger(mpath, cfg, test_features, name_override=name)
        if trigger is not None:
            triggers[name] = {uid: trigger.score(uid) for uid in utt_ids
                              if uid in test_features}
            print(f"  Loaded {name}")

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
            "n_utterances": len(trig_scores),
        }

        for k, v in summary.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

    # Paired bootstrap comparisons
    paired_tests = []
    if "probe_gain" in triggers:
        if "argmax_accent" in triggers:
            paired_tests.append(("probe_gain", "argmax_accent"))
        if "scalar_probe" in triggers:
            paired_tests.append(("probe_gain", "scalar_probe"))
    if "probe_capped_wer" in triggers and "scalar_probe" in triggers:
        paired_tests.append(("probe_capped_wer", "scalar_probe"))

    results["paired_bootstrap"] = {}
    for a_name, b_name in paired_tests:
        print(f"\nPaired bootstrap: {a_name} vs {b_name}")
        paired = paired_bootstrap(
            triggers[a_name], triggers[b_name],
            default_wers, careful_wers,
            n=args.bootstrap_n, seed=cfg.get("seed", 42),
            random_scores=random_scores,
        )
        results["paired_bootstrap"][f"{a_name}_vs_{b_name}"] = paired

        # Print key results
        for metric, diff_data in paired["diffs"].items():
            sig = "***" if diff_data["significant"] else ""
            print(f"  {metric}: {diff_data['mean_diff']:.4f} "
                  f"[{diff_data['ci_lo']:.4f}, {diff_data['ci_hi']:.4f}] {sig}")

    # Save
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
