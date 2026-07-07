"""EXP-11: Temporal std evaluation.

Loads WavLM stats features (25, 2048) — split into mean (25, 1024) and std (25, 1024).
Computes scalar "temporal variability" = mean of per-layer std (25 values -> 1 scalar).
Folds into Phase 2 combiner as additional feature, re-fits with and without temporal_std.

Subset analysis: evaluates on confidently-hallucinated utterances (capped WER > 0 AND
avg_logprob > median).

Produces: experiments/EXP-11-temporal-std/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import (
    bootstrap,
    cap_wer,
    combiner_eval,
    decision_scorecard,
    escalation_gain,
    operating_curve,
    paired_bootstrap,
    summarize,
)
from accentedness_routing.triggers.confidence import ConfidenceTrigger
from accentedness_routing.triggers.hallucination import NoSpeechProbTrigger
from accentedness_routing.triggers.multitask_probe import MultiTaskProbe
from accentedness_routing.triggers.random_trigger import RandomTrigger


def _model_slug(model_path: str) -> str:
    return model_path.split("/")[-1]


def load_temporal_std(stats_dir: Path, utterance_id: str) -> float | None:
    """Load stats features and compute temporal variability scalar.

    Stats features are (25, 2048) = [mean(25, 1024) | std(25, 1024)].
    Temporal variability = mean of per-layer std values.
    """
    feat_path = stats_dir / f"{utterance_id}.pt"
    if not feat_path.exists():
        return None

    stats = torch.load(feat_path, weights_only=True)  # (25, 2048)
    # std portion is the second half
    std_part = stats[:, stats.shape[1] // 2:]  # (25, 1024)
    # Mean of per-layer std: average across features, then across layers
    temporal_std = float(std_part.mean())
    return temporal_std


def build_feature_matrix(
    utterances,
    cache_dir: str,
    default_model: str,
    careful_model: str,
    features_dir: Path,
    acoustic_dir: Path,
    stats_dir: Path,
    champion_model,
    champion_cal,
    include_temporal: bool = True,
):
    """Build feature matrix for a split.

    Without temporal: [confidence, no_speech_prob, champion_score, duration, silence_ratio, speaking_rate]
    With temporal: [confidence, no_speech_prob, champion_score, duration, silence_ratio, speaking_rate, temporal_std]
    """
    features_list = []
    gains_list = []
    utt_ids = []
    logprobs = []

    for utt in utterances:
        uid = utt.utterance_id
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d is None or c is None:
            continue

        # Acoustic features
        acoustic_path = acoustic_dir / f"{uid}.json"
        if not acoustic_path.exists():
            continue

        with open(acoustic_path) as f:
            acoustic = json.load(f)

        # Temporal std
        if include_temporal:
            temporal_std = load_temporal_std(stats_dir, uid)
            if temporal_std is None:
                continue

        # Champion score
        champion_score = 0.5
        if champion_model is not None and champion_cal is not None:
            feat_path = features_dir / f"{uid}.pt"
            if feat_path.exists():
                feat = torch.load(feat_path, weights_only=True)
                with torch.no_grad():
                    raw = champion_model.predict_score(feat.unsqueeze(0)).item()
                low, high = champion_cal["low"], champion_cal["high"]
                rng = high - low if high - low > 1e-8 else 1.0
                champion_score = max(0.0, min(1.0, (raw - low) / rng))

        avg_logprob = d.get("avg_logprob")
        no_speech_prob = d.get("no_speech_prob")
        if avg_logprob is None or (isinstance(avg_logprob, float) and np.isnan(avg_logprob)):
            avg_logprob = 0.0
        if no_speech_prob is None or (isinstance(no_speech_prob, float) and np.isnan(no_speech_prob)):
            no_speech_prob = 0.0

        feature_vec = [
            avg_logprob,
            no_speech_prob,
            champion_score,
            acoustic["duration"],
            acoustic["silence_ratio"],
            acoustic["speaking_rate"],
        ]
        if include_temporal:
            feature_vec.append(temporal_std)

        features_list.append(np.array(feature_vec))
        gains_list.append(escalation_gain(d["wer"], c["wer"]))
        utt_ids.append(uid)
        logprobs.append(avg_logprob)

    if not features_list:
        return np.array([]), np.array([]), [], np.array([])

    return np.array(features_list), np.array(gains_list), utt_ids, np.array(logprobs)


def normalize_features(train_X, val_X, test_X):
    """Z-score normalize using train statistics."""
    if len(train_X) == 0:
        return train_X, val_X, test_X
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_X - mean) / std, (val_X - mean) / std, (test_X - mean) / std


def fit_and_evaluate(
    train_X, train_gains, val_X, val_gains, test_X, test_gains,
    test_ids, test_wer_default, test_wer_careful,
    confidence_scores, random_scores,
    tau, bootstrap_n, seed, label,
):
    """Fit logistic combiner and evaluate."""
    y_val = (val_gains > tau).astype(int)
    y_test = (test_gains > tau).astype(int)

    if len(np.unique(y_val)) < 2:
        print(f"  {label}: single class in val, skipping")
        return None

    # Combine train+val for fitting (more data)
    fit_X = np.vstack([train_X, val_X])
    fit_y = np.concatenate([(train_gains > tau).astype(int), y_val])

    if len(np.unique(fit_y)) < 2:
        print(f"  {label}: single class in train+val, skipping")
        return None

    model = LogisticRegressionCV(
        Cs=10,
        l1_ratios=(0,),
        cv=5,
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(fit_X, fit_y)

    # Val AUC (on val portion only)
    val_proba = model.predict_proba(val_X)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_proba))

    # Test
    test_proba = model.predict_proba(test_X)[:, 1]
    combiner_scores = {uid: float(test_proba[i]) for i, uid in enumerate(test_ids)}

    test_auc = None
    if len(np.unique(y_test)) >= 2:
        test_auc = float(roc_auc_score(y_test, test_proba))

    # Full eval pipeline
    curve = operating_curve(combiner_scores, test_wer_default, test_wer_careful)
    random_curve = operating_curve(random_scores, test_wer_default, test_wer_careful)
    summary = summarize(curve, random_curve)
    test_gains_dict = {uid: escalation_gain(test_wer_default[uid], test_wer_careful[uid])
                       for uid in test_ids}

    boot = bootstrap(combiner_scores, test_wer_default, test_wer_careful,
                      n=bootstrap_n, seed=seed, random_scores=random_scores)

    paired = paired_bootstrap(
        combiner_scores, confidence_scores,
        test_wer_default, test_wer_careful,
        n=bootstrap_n, seed=seed, random_scores=random_scores,
    )

    print(f"  {label}: val_auc={val_auc:.3f}, test_auc={test_auc}")
    for k, v in summary.items():
        if v is not None:
            print(f"    {k}: {v:.4f}")

    return {
        "val_auc": val_auc,
        "test_auc": test_auc,
        "coef": model.coef_.tolist(),
        "summary": summary,
        "scorecard_tau_0.00": decision_scorecard(combiner_scores, test_gains_dict, 0.0),
        "scorecard_tau_0.05": decision_scorecard(combiner_scores, test_gains_dict, tau),
        "bootstrap_ci": boot["ci"],
        "curve_bands": boot["curve_bands"],
        "paired_vs_confidence": paired,
        "combiner_scores": combiner_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-11: Temporal std evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    cache_dir = cfg["asr"]["cache_dir"]
    features_dir = Path(cfg["features"]["cache_dir"])
    stats_dir = Path(cfg.get("features_stats", {}).get("cache_dir", "data/features_cache_stats"))
    acoustic_dir = Path(cfg.get("acoustic_features", {}).get(
        "cache_dir", "data/acoustic_features_cache"))
    models_dir = Path("models")

    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-11-temporal-std"
    exp_dir.mkdir(parents=True, exist_ok=True)

    combiner_cfg = cfg.get("combiner", {})
    tau = combiner_cfg.get("tau", 0.05)
    seed = cfg.get("seed", 42)

    # Use the widest-gap regime for temporal analysis
    default_model = "mlx-community/whisper-tiny-mlx"
    careful_model = "mlx-community/whisper-large-v3-mlx"
    slug = f"{_model_slug(default_model)}__{_model_slug(careful_model)}"

    # Load splits
    splits = {}
    for name in ["train", "val", "test"]:
        pkl_path = data_dir / f"{name}_utterances.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                splits[name] = pickle.load(f)
            print(f"Loaded {len(splits[name])} {name} utterances")

    # Load champion model
    champion_path = models_dir / f"champion_{slug}.pt"
    champion_model = None
    champion_cal = None
    if champion_path.exists():
        checkpoint = torch.load(champion_path, weights_only=False)
        accent_to_idx = checkpoint.get("accent_to_idx", {})
        champion_model = MultiTaskProbe(
            num_accents=len(accent_to_idx) if accent_to_idx else 6,
            num_layers=cfg["features"]["num_layers"],
            hidden_dim=cfg["features"]["hidden_dim"],
            probe_dim=cfg.get("probe_gain", cfg["probe"])["hidden_dim"],
            dropout=cfg.get("probe_gain", cfg["probe"])["dropout"],
        )
        champion_model.load_state_dict(checkpoint["model_state_dict"])
        champion_model.eval()
        champion_cal = checkpoint["calibration"]

    print(f"\nBuilding features WITH temporal_std...")
    train_X_with, train_gains, train_ids, _ = build_feature_matrix(
        splits.get("train", []), cache_dir, default_model, careful_model,
        features_dir, acoustic_dir, stats_dir, champion_model, champion_cal,
        include_temporal=True)
    val_X_with, val_gains, val_ids, _ = build_feature_matrix(
        splits.get("val", []), cache_dir, default_model, careful_model,
        features_dir, acoustic_dir, stats_dir, champion_model, champion_cal,
        include_temporal=True)
    test_X_with, test_gains, test_ids, test_logprobs = build_feature_matrix(
        splits.get("test", []), cache_dir, default_model, careful_model,
        features_dir, acoustic_dir, stats_dir, champion_model, champion_cal,
        include_temporal=True)

    print(f"  WITH temporal: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    print(f"\nBuilding features WITHOUT temporal_std...")
    train_X_without, train_gains_wo, _, _ = build_feature_matrix(
        splits.get("train", []), cache_dir, default_model, careful_model,
        features_dir, acoustic_dir, stats_dir, champion_model, champion_cal,
        include_temporal=False)
    val_X_without, val_gains_wo, _, _ = build_feature_matrix(
        splits.get("val", []), cache_dir, default_model, careful_model,
        features_dir, acoustic_dir, stats_dir, champion_model, champion_cal,
        include_temporal=False)
    test_X_without, test_gains_wo, test_ids_wo, _ = build_feature_matrix(
        splits.get("test", []), cache_dir, default_model, careful_model,
        features_dir, acoustic_dir, stats_dir, champion_model, champion_cal,
        include_temporal=False)

    if len(test_ids) == 0:
        print("No test data with stats features. Run extract_features_stats.py first.")
        return

    # Normalize
    train_X_with, val_X_with, test_X_with = normalize_features(
        train_X_with, val_X_with, test_X_with)
    train_X_without, val_X_without, test_X_without = normalize_features(
        train_X_without, val_X_without, test_X_without)

    # Build reference dicts
    test_wer_default = {}
    test_wer_careful = {}
    for utt in splits.get("test", []):
        uid = utt.utterance_id
        if uid not in test_ids:
            continue
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d and c:
            test_wer_default[uid] = d["wer"]
            test_wer_careful[uid] = c["wer"]

    test_logprobs_dict = {}
    for uid in test_ids:
        d = load_cached(cache_dir, default_model, uid)
        if d is not None:
            lp = d.get("avg_logprob")
            if lp is None or (isinstance(lp, float) and np.isnan(lp)):
                lp = 0.0
            test_logprobs_dict[uid] = lp
    confidence_trigger = ConfidenceTrigger(test_logprobs_dict)
    confidence_scores = {uid: confidence_trigger.score(uid) for uid in test_ids}
    random_trigger = RandomTrigger(seed=seed)
    random_scores = {uid: random_trigger.score(uid) for uid in test_ids}

    results = {"pairing": slug, "regime": "wide", "triggers": {}}

    # Fit combiner WITH temporal_std
    print("\n--- Combiner WITH temporal_std ---")
    result_with = fit_and_evaluate(
        train_X_with, train_gains, val_X_with, val_gains,
        test_X_with, test_gains, test_ids,
        test_wer_default, test_wer_careful,
        confidence_scores, random_scores,
        tau, args.bootstrap_n, seed, "with_temporal")

    # Fit combiner WITHOUT temporal_std
    print("\n--- Combiner WITHOUT temporal_std ---")
    result_without = fit_and_evaluate(
        train_X_without, train_gains_wo, val_X_without, val_gains_wo,
        test_X_without, test_gains_wo, test_ids_wo,
        test_wer_default, test_wer_careful,
        confidence_scores, random_scores,
        tau, args.bootstrap_n, seed, "without_temporal")

    if result_with:
        results["triggers"]["combiner_with_temporal"] = {
            k: v for k, v in result_with.items() if k != "combiner_scores"
        }
    if result_without:
        results["triggers"]["combiner_without_temporal"] = {
            k: v for k, v in result_without.items() if k != "combiner_scores"
        }

    # Paired bootstrap: with vs without temporal
    if result_with and result_without:
        print("\n--- Paired: with_temporal vs without_temporal ---")
        # Use common test IDs
        common_ids = sorted(set(test_ids) & set(test_ids_wo))
        scores_with = {uid: result_with["combiner_scores"][uid] for uid in common_ids
                       if uid in result_with["combiner_scores"]}
        scores_without = {uid: result_without["combiner_scores"][uid] for uid in common_ids
                          if uid in result_without["combiner_scores"]}

        paired = paired_bootstrap(
            scores_with, scores_without,
            test_wer_default, test_wer_careful,
            n=args.bootstrap_n, seed=seed,
            random_scores=random_scores,
        )
        results["paired_with_vs_without_temporal"] = paired

        avr_diff = paired.get("diffs", {}).get("area_vs_random", {})
        print(f"  Area vs random diff: {avr_diff.get('mean_diff', 'N/A'):.4f} "
              f"[{avr_diff.get('ci_lo', 'N/A'):.4f}, {avr_diff.get('ci_hi', 'N/A'):.4f}] "
              f"sig={avr_diff.get('significant', 'N/A')}")

    # Subset analysis: confidently-hallucinated utterances
    print("\n--- Subset: confidently-hallucinated ---")
    median_logprob = float(np.median(test_logprobs)) if len(test_logprobs) > 0 else 0.0
    conf_hall_ids = [
        uid for i, uid in enumerate(test_ids)
        if cap_wer(test_wer_default.get(uid, 0)) > 0
        and test_logprobs[i] > median_logprob
    ]
    print(f"  Confidently-hallucinated subset: {len(conf_hall_ids)} / {len(test_ids)} utterances")

    if len(conf_hall_ids) >= 10 and result_with and result_without:
        # Evaluate both combiners on this subset
        subset_wer_d = {uid: test_wer_default[uid] for uid in conf_hall_ids
                        if uid in test_wer_default}
        subset_wer_c = {uid: test_wer_careful[uid] for uid in conf_hall_ids
                        if uid in test_wer_careful}
        subset_random = {uid: random_scores[uid] for uid in conf_hall_ids
                         if uid in random_scores}
        subset_conf = {uid: confidence_scores[uid] for uid in conf_hall_ids
                       if uid in confidence_scores}

        for label, r in [("with_temporal", result_with), ("without_temporal", result_without)]:
            subset_scores = {uid: r["combiner_scores"][uid] for uid in conf_hall_ids
                             if uid in r["combiner_scores"]}
            if len(subset_scores) < 5:
                continue

            curve = operating_curve(subset_scores, subset_wer_d, subset_wer_c)
            random_curve = operating_curve(subset_random, subset_wer_d, subset_wer_c)
            summary = summarize(curve, random_curve)

            results[f"subset_conf_hall_{label}"] = {
                "n_utterances": len(subset_scores),
                "summary": summary,
            }
            print(f"  {label} subset area_vs_random: "
                  f"{summary.get('area_vs_random', 'N/A')}")

    # Save
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
