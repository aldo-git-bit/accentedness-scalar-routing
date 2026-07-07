"""EXP-12: Accent-adapted careful path evaluation.

Verify-before-trust guard:
  Compare adapted model's WER on Indian English test slice against
  large-v3 and turbo. If adapted doesn't clearly beat general large-v3
  on Indian English -> report negative, stop.

If guard passes:
  Define careful path for Indian slice as adapted model; rest use large-v3.
  Compare headroom for general vs accent-adapted careful path on Indian slice.

Produces: experiments/EXP-12-accent-adapted/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import (
    bootstrap,
    cap_wer,
    decision_scorecard,
    escalation_gain,
    headroom_summary,
    operating_curve,
    paired_bootstrap,
    summarize,
)
from accentedness_routing.triggers.confidence import ConfidenceTrigger
from accentedness_routing.triggers.oracle import OracleTrigger
from accentedness_routing.triggers.random_trigger import RandomTrigger


def main():
    parser = argparse.ArgumentParser(description="EXP-12: Accent-adapted evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    cache_dir = cfg["asr"]["cache_dir"]
    seed = cfg.get("seed", 42)

    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-12-accent-adapted"
    exp_dir.mkdir(parents=True, exist_ok=True)

    adapted_model = "Tejveer12/Indian-Accent-English-Whisper-Finetuned"
    general_large_v3 = "mlx-community/whisper-large-v3-mlx"
    turbo = "mlx-community/whisper-large-v3-turbo"
    default_model = "mlx-community/whisper-tiny-mlx"  # widest gap default

    # Load test utterances
    with open(data_dir / "test_utterances.pkl", "rb") as f:
        test_utts = pickle.load(f)
    print(f"Loaded {len(test_utts)} test utterances")

    # Indian English subset
    indian_utts = [utt for utt in test_utts if utt.accent == "indian"]
    print(f"Indian English test subset: {len(indian_utts)} utterances")

    if len(indian_utts) == 0:
        print("No Indian English utterances found. Stopping.")
        return

    # -----------------------------------------------------------------------
    # Guard: verify adapted model beats general large-v3 on Indian English
    # -----------------------------------------------------------------------
    print("\n--- Verify-before-trust guard ---")

    models_to_check = {
        "adapted": adapted_model,
        "large_v3": general_large_v3,
        "turbo": turbo,
    }

    guard_wers = {}
    for label, model_id in models_to_check.items():
        wers = []
        n_found = 0
        for utt in indian_utts:
            result = load_cached(cache_dir, model_id, utt.utterance_id)
            if result is not None:
                wers.append(cap_wer(result["wer"]))
                n_found += 1
        mean_wer = float(np.mean(wers)) if wers else None
        guard_wers[label] = {
            "mean_capped_wer": mean_wer,
            "n_found": n_found,
            "n_total": len(indian_utts),
        }
        print(f"  {label}: mean capped WER = {mean_wer:.4f} ({n_found}/{len(indian_utts)} found)"
              if mean_wer is not None else f"  {label}: no cached results")

    results = {"guard": guard_wers}

    # Guard check: adapted must beat large-v3 on Indian English
    adapted_wer = guard_wers.get("adapted", {}).get("mean_capped_wer")
    large_v3_wer = guard_wers.get("large_v3", {}).get("mean_capped_wer")

    if adapted_wer is None:
        print("\nGuard FAIL: No adapted model results. Run run_asr_adapted.py first.")
        results["guard_passed"] = False
        results["guard_reason"] = "no_adapted_results"
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    if large_v3_wer is None:
        print("\nGuard FAIL: No large-v3 results for Indian English.")
        results["guard_passed"] = False
        results["guard_reason"] = "no_large_v3_results"
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # Guard passes if adapted WER < large-v3 WER (clearly better)
    improvement = large_v3_wer - adapted_wer
    guard_passed = improvement > 0.01  # At least 1% absolute improvement
    results["guard_passed"] = guard_passed
    results["guard_improvement"] = improvement

    if not guard_passed:
        print(f"\nGuard FAIL: Adapted model does not clearly beat general large-v3 "
              f"on Indian English (improvement={improvement:.4f})")
        results["guard_reason"] = "insufficient_improvement"
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {exp_dir / 'metrics.json'}")
        return

    print(f"\nGuard PASS: Adapted model improves by {improvement:.4f} on Indian English")

    # -----------------------------------------------------------------------
    # Compare headroom: general vs accent-adapted careful path on Indian slice
    # -----------------------------------------------------------------------
    print("\n--- Headroom comparison on Indian slice ---")

    # Default model WERs (for Indian slice)
    default_wers = {}
    default_logprobs = {}
    for utt in indian_utts:
        uid = utt.utterance_id
        d = load_cached(cache_dir, default_model, uid)
        if d is not None:
            default_wers[uid] = d["wer"]
            default_logprobs[uid] = d.get("avg_logprob", 0.0)

    # General large-v3 careful WERs
    general_careful_wers = {}
    for utt in indian_utts:
        uid = utt.utterance_id
        c = load_cached(cache_dir, general_large_v3, uid)
        if c is not None:
            general_careful_wers[uid] = c["wer"]

    # Adapted careful WERs
    adapted_careful_wers = {}
    for utt in indian_utts:
        uid = utt.utterance_id
        c = load_cached(cache_dir, adapted_model, uid)
        if c is not None:
            adapted_careful_wers[uid] = c["wer"]

    # Common IDs for general path
    general_ids = sorted(
        set(default_wers.keys()) & set(general_careful_wers.keys()))
    # Common IDs for adapted path
    adapted_ids = sorted(
        set(default_wers.keys()) & set(adapted_careful_wers.keys()))

    print(f"  General path: {len(general_ids)} utterances")
    print(f"  Adapted path: {len(adapted_ids)} utterances")

    # Evaluate both paths
    for label, careful_wers, common_ids in [
        ("general_careful", general_careful_wers, general_ids),
        ("adapted_careful", adapted_careful_wers, adapted_ids),
    ]:
        if len(common_ids) < 5:
            print(f"  {label}: insufficient data, skipping")
            continue

        wd = {uid: default_wers[uid] for uid in common_ids}
        wc = {uid: careful_wers[uid] for uid in common_ids}

        oracle = OracleTrigger(wd, wc)
        random_trigger = RandomTrigger(seed=seed)

        oracle_scores = {uid: oracle.score(uid) for uid in common_ids}
        random_scores = {uid: random_trigger.score(uid) for uid in common_ids}

        logprobs_subset = {uid: default_logprobs[uid] for uid in common_ids
                          if uid in default_logprobs}
        if logprobs_subset:
            confidence = ConfidenceTrigger(logprobs_subset)
            confidence_scores = {uid: confidence.score(uid) for uid in common_ids
                                 if uid in logprobs_subset}
        else:
            confidence_scores = random_scores

        headroom = headroom_summary(wd, wc, oracle_scores, random_scores)

        # Full eval for oracle and confidence
        triggers_data = {}
        for trig_name, trig_scores in [("oracle", oracle_scores),
                                        ("confidence", confidence_scores),
                                        ("random", random_scores)]:
            curve = operating_curve(trig_scores, wd, wc)
            random_curve = operating_curve(random_scores, wd, wc)
            summary = summarize(curve, random_curve)
            gains = {uid: escalation_gain(wd[uid], wc[uid]) for uid in common_ids}

            boot = bootstrap(trig_scores, wd, wc, n=args.bootstrap_n, seed=seed,
                             random_scores=random_scores)

            triggers_data[trig_name] = {
                "summary": summary,
                "scorecard_tau_0.00": decision_scorecard(trig_scores, gains, 0.0),
                "scorecard_tau_0.05": decision_scorecard(trig_scores, gains, 0.05),
                "bootstrap_ci": boot["ci"],
                "curve_bands": boot["curve_bands"],
            }

        results[f"{label}_path"] = {
            "headroom": headroom,
            "triggers": triggers_data,
            "n_utterances": len(common_ids),
        }

        print(f"  {label}: oracle_avr={headroom['oracle_area_vs_random']:.4f}, "
              f"wer_gap={headroom['wer_gap']:.4f}")

    # Save
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
