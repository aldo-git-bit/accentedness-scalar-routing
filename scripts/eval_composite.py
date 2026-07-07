"""EXP-10: Composite trigger evaluation.

Combines confidence, no_speech_prob, champion score, and acoustic features
via L2 logistic regression. Tests under two regimes from EXP-09:
  - Narrow gap: small -> large-v3 (Round 2 baseline)
  - Widest gap: tiny -> large-v3

If logistic val AUC < 0.55, also tries GBM as fallback.

Produces: experiments/EXP-10-composite/{regime_slug}/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.ensemble import GradientBoostingClassifier
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


def load_features_for_split(
    utterances,
    cache_dir: str,
    default_model: str,
    careful_model: str,
    features_dir: Path,
    acoustic_dir: Path,
    champion_model: MultiTaskProbe | None,
    champion_cal: dict | None,
):
    """Build feature matrix and gain vector for a split.

    Feature vector per utterance:
    [confidence, no_speech_prob, champion_score, duration, silence_ratio, speaking_rate]
    """
    features_list = []
    gains_list = []
    utt_ids = []

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

        # Champion score
        champion_score = 0.5  # default
        if champion_model is not None and champion_cal is not None:
            feat_path = features_dir / f"{uid}.pt"
            if feat_path.exists():
                feat = torch.load(feat_path, weights_only=True)
                with torch.no_grad():
                    raw = champion_model.predict_score(feat.unsqueeze(0)).item()
                low, high = champion_cal["low"], champion_cal["high"]
                rng = high - low if high - low > 1e-8 else 1.0
                champion_score = max(0.0, min(1.0, (raw - low) / rng))

        # Build feature vector
        # Confidence: normalized 1 - logprob (higher = less confident = escalate)
        # We'll normalize globally after collecting all data
        avg_logprob = d.get("avg_logprob")
        no_speech_prob = d.get("no_speech_prob")
        if avg_logprob is None or (isinstance(avg_logprob, float) and np.isnan(avg_logprob)):
            avg_logprob = 0.0
        if no_speech_prob is None or (isinstance(no_speech_prob, float) and np.isnan(no_speech_prob)):
            no_speech_prob = 0.0
        feature_vec = np.array([
            avg_logprob,
            no_speech_prob,
            champion_score,
            acoustic["duration"],
            acoustic["silence_ratio"],
            acoustic["speaking_rate"],
        ])

        gain = escalation_gain(d["wer"], c["wer"])

        features_list.append(feature_vec)
        gains_list.append(gain)
        utt_ids.append(uid)

    if not features_list:
        return np.array([]), np.array([]), []

    features_matrix = np.array(features_list)
    gains_arr = np.array(gains_list)

    return features_matrix, gains_arr, utt_ids


def normalize_features(train_X, val_X, test_X):
    """Z-score normalize using train statistics."""
    if len(train_X) == 0:
        return train_X, val_X, test_X

    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std < 1e-8] = 1.0

    return (train_X - mean) / std, (val_X - mean) / std, (test_X - mean) / std


def main():
    parser = argparse.ArgumentParser(description="EXP-10: Composite trigger evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    cache_dir = cfg["asr"]["cache_dir"]
    features_dir = Path(cfg["features"]["cache_dir"])
    acoustic_dir = Path(cfg.get("acoustic_features", {}).get(
        "cache_dir", "data/acoustic_features_cache"))
    models_dir = Path("models")

    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-10-composite"
    exp_dir.mkdir(parents=True, exist_ok=True)

    combiner_cfg = cfg.get("combiner", {})
    tau = combiner_cfg.get("tau", 0.05)
    seed = cfg.get("seed", 42)

    # Load splits
    splits = {}
    for name in ["train", "val", "test"]:
        pkl_path = data_dir / f"{name}_utterances.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                splits[name] = pickle.load(f)
            print(f"Loaded {len(splits[name])} {name} utterances")

    # Define regimes
    regimes = [
        {
            "name": "narrow",
            "default": "mlx-community/whisper-small-mlx",
            "careful": "mlx-community/whisper-large-v3-mlx",
        },
        {
            "name": "wide",
            "default": "mlx-community/whisper-tiny-mlx",
            "careful": "mlx-community/whisper-large-v3-mlx",
        },
    ]

    for regime in regimes:
        default_model = regime["default"]
        careful_model = regime["careful"]
        regime_name = regime["name"]
        slug = f"{_model_slug(default_model)}__{_model_slug(careful_model)}"

        print(f"\n{'='*60}")
        print(f"Regime: {regime_name} ({slug})")
        print(f"{'='*60}")

        regime_dir = exp_dir / regime_name
        regime_dir.mkdir(parents=True, exist_ok=True)

        # Load champion model for this pairing (if retrained in Phase 1)
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
            print(f"  Loaded champion from {champion_path}")
        else:
            print(f"  No champion model at {champion_path}, using default score=0.5")

        # Build feature matrices for train, val, test
        train_X, train_gains, train_ids = load_features_for_split(
            splits.get("train", []), cache_dir, default_model, careful_model,
            features_dir, acoustic_dir, champion_model, champion_cal)
        val_X, val_gains, val_ids = load_features_for_split(
            splits.get("val", []), cache_dir, default_model, careful_model,
            features_dir, acoustic_dir, champion_model, champion_cal)
        test_X, test_gains, test_ids = load_features_for_split(
            splits.get("test", []), cache_dir, default_model, careful_model,
            features_dir, acoustic_dir, champion_model, champion_cal)

        print(f"  Features: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        if len(val_ids) == 0 or len(test_ids) == 0:
            print(f"  SKIP: insufficient data")
            continue

        # Normalize
        train_X, val_X, test_X = normalize_features(train_X, val_X, test_X)

        # Build confidence and random scores for comparison
        test_logprobs = {}
        test_wer_default = {}
        test_wer_careful = {}
        for utt in splits.get("test", []):
            uid = utt.utterance_id
            if uid not in test_ids:
                continue
            d = load_cached(cache_dir, default_model, uid)
            c = load_cached(cache_dir, careful_model, uid)
            if d and c:
                test_logprobs[uid] = d.get("avg_logprob", 0.0)
                test_wer_default[uid] = d["wer"]
                test_wer_careful[uid] = c["wer"]

        confidence_trigger = ConfidenceTrigger(test_logprobs)
        confidence_scores = {uid: confidence_trigger.score(uid) for uid in test_ids}

        random_trigger = RandomTrigger(seed=seed)
        random_scores = {uid: random_trigger.score(uid) for uid in test_ids}

        # Run combiner_eval
        result = combiner_eval(
            val_X, val_gains,
            test_X, test_gains,
            confidence_scores, test_wer_default, test_wer_careful, random_scores,
            tau=tau, bootstrap_n=args.bootstrap_n, seed=seed,
        )

        family_used = "logistic"

        # GBM fallback if logistic val AUC < 0.55
        if "error" not in result and result.get("val_auc", 0) < 0.55:
            print(f"  Logistic val AUC={result['val_auc']:.3f} < 0.55, trying GBM...")

            y_val = (val_gains > tau).astype(int)
            y_test = (test_gains > tau).astype(int)

            if len(np.unique(y_val)) >= 2:
                gbm = GradientBoostingClassifier(
                    max_depth=2, n_estimators=50, random_state=seed)
                gbm.fit(val_X, y_val)
                gbm_val_proba = gbm.predict_proba(val_X)[:, 1]
                gbm_val_auc = float(roc_auc_score(y_val, gbm_val_proba))

                if gbm_val_auc > result.get("val_auc", 0):
                    print(f"  GBM val AUC={gbm_val_auc:.3f} > logistic, using GBM")
                    gbm_test_proba = gbm.predict_proba(test_X)[:, 1]
                    gbm_scores = {uid: float(gbm_test_proba[i])
                                  for i, uid in enumerate(test_ids)}

                    gbm_test_auc = None
                    if len(np.unique(y_test)) >= 2:
                        gbm_test_auc = float(roc_auc_score(y_test, gbm_test_proba))

                    paired = paired_bootstrap(
                        gbm_scores, confidence_scores,
                        test_wer_default, test_wer_careful,
                        n=args.bootstrap_n, seed=seed,
                        random_scores=random_scores,
                    )

                    result = {
                        "combiner_scores": gbm_scores,
                        "val_auc": gbm_val_auc,
                        "test_auc": gbm_test_auc,
                        "paired_vs_confidence": paired,
                        "family": "gbm",
                        "logistic_val_auc": result.get("val_auc"),
                    }
                    family_used = "gbm"

        print(f"  Family: {family_used}")
        if "error" not in result:
            print(f"  Val AUC: {result.get('val_auc', 'N/A')}")
            print(f"  Test AUC: {result.get('test_auc', 'N/A')}")

        # Also run confidence through full pipeline for metrics.json format
        triggers_results = {"triggers": {}, "pairing": slug, "regime": regime_name}

        # Confidence
        conf_curve = operating_curve(confidence_scores, test_wer_default, test_wer_careful)
        random_curve = operating_curve(random_scores, test_wer_default, test_wer_careful)
        test_gains_dict = {uid: escalation_gain(test_wer_default[uid], test_wer_careful[uid])
                          for uid in test_ids}

        conf_summary = summarize(conf_curve, random_curve)
        conf_boot = bootstrap(confidence_scores, test_wer_default, test_wer_careful,
                              n=args.bootstrap_n, seed=seed, random_scores=random_scores)

        triggers_results["triggers"]["confidence"] = {
            "summary": conf_summary,
            "scorecard_tau_0.00": decision_scorecard(confidence_scores, test_gains_dict, 0.0),
            "scorecard_tau_0.05": decision_scorecard(confidence_scores, test_gains_dict, 0.05),
            "bootstrap_ci": conf_boot["ci"],
            "curve_bands": conf_boot["curve_bands"],
        }

        # Combiner
        if "error" not in result and "combiner_scores" in result:
            combiner_scores = result["combiner_scores"]
            comb_curve = operating_curve(combiner_scores, test_wer_default, test_wer_careful)
            comb_summary = summarize(comb_curve, random_curve)
            comb_boot = bootstrap(combiner_scores, test_wer_default, test_wer_careful,
                                  n=args.bootstrap_n, seed=seed, random_scores=random_scores)

            triggers_results["triggers"]["combiner"] = {
                "summary": comb_summary,
                "scorecard_tau_0.00": decision_scorecard(
                    combiner_scores, test_gains_dict, 0.0),
                "scorecard_tau_0.05": decision_scorecard(
                    combiner_scores, test_gains_dict, 0.05),
                "bootstrap_ci": comb_boot["ci"],
                "curve_bands": comb_boot["curve_bands"],
            }

        triggers_results["combiner_details"] = {
            k: v for k, v in result.items()
            if k != "combiner_scores"
        }

        with open(regime_dir / "metrics.json", "w") as f:
            json.dump(triggers_results, f, indent=2, default=str)
        print(f"  Saved: {regime_dir / 'metrics.json'}")

    print(f"\nComposite evaluation complete. Results in {exp_dir}")


if __name__ == "__main__":
    main()
