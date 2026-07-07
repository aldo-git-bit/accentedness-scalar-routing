"""Canonical evaluation module for Rounds 2 and 3.

All Round 2+ scripts import exclusively from here. Pilot eval code
(routing/router.py, routing/metrics.py, eval/curves.py) remains
untouched for backward compatibility.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def cap_wer(w: float) -> float:
    """Cap WER at 1.0 to neutralize hallucination artifacts."""
    return min(w, 1.0)


def escalation_gain(wer_default: float, wer_careful: float) -> float:
    """Compute escalation gain with capped WER.

    gain = cap(default) - cap(careful).  Can be negative if careful is worse.
    """
    return cap_wer(wer_default) - cap_wer(wer_careful)


# ---------------------------------------------------------------------------
# Operating curve
# ---------------------------------------------------------------------------

def operating_curve(
    scores: dict[str, float],
    wer_default: dict[str, float],
    wer_careful: dict[str, float],
    aggregation: str = "micro",
    accent_map: dict[str, str] | None = None,
    num_thresholds: int = 101,
) -> dict:
    """Threshold sweep with capped WER.

    Args:
        scores: utterance_id -> routing score in [0, 1]. Higher = escalate.
        wer_default: utterance_id -> default model WER (uncapped).
        wer_careful: utterance_id -> careful model WER (uncapped).
        aggregation: "micro" (global average) or "macro" (per-accent then average).
        accent_map: required for macro aggregation. utterance_id -> accent label.
        num_thresholds: number of threshold points in [0, 1].

    Returns:
        dict with keys: thresholds, escalation_rates, net_wers.
    """
    if aggregation == "macro" and accent_map is None:
        raise ValueError("accent_map required for macro aggregation")

    utt_ids = sorted(scores.keys())
    thresholds = np.linspace(0, 1, num_thresholds).tolist()

    # Pre-cap all WERs
    capped_default = {uid: cap_wer(wer_default[uid]) for uid in utt_ids}
    capped_careful = {uid: cap_wer(wer_careful[uid]) for uid in utt_ids}

    escalation_rates = []
    net_wers = []

    for thresh in thresholds:
        escalated = set(uid for uid in utt_ids if scores[uid] >= thresh)
        esc_rate = len(escalated) / len(utt_ids)

        if aggregation == "micro":
            wer_sum = 0.0
            for uid in utt_ids:
                if uid in escalated:
                    wer_sum += capped_careful[uid]
                else:
                    wer_sum += capped_default[uid]
            net_wer = wer_sum / len(utt_ids)
        else:
            # macro: per-accent mean, then average across accents
            accent_groups: dict[str, list[str]] = defaultdict(list)
            for uid in utt_ids:
                accent_groups[accent_map[uid]].append(uid)

            accent_wers = []
            for _accent, group_uids in sorted(accent_groups.items()):
                group_wer_sum = 0.0
                for uid in group_uids:
                    if uid in escalated:
                        group_wer_sum += capped_careful[uid]
                    else:
                        group_wer_sum += capped_default[uid]
                accent_wers.append(group_wer_sum / len(group_uids))
            net_wer = float(np.mean(accent_wers))

        escalation_rates.append(esc_rate)
        net_wers.append(net_wer)

    return {
        "thresholds": thresholds,
        "escalation_rates": escalation_rates,
        "net_wers": net_wers,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _interp_net_wer(curve: dict, budget: float) -> float | None:
    """Interpolate net WER at a given escalation budget."""
    rates = np.array(curve["escalation_rates"])
    wers = np.array(curve["net_wers"])
    order = np.argsort(rates)
    rates_sorted = rates[order]
    wers_sorted = wers[order]

    if budget < rates_sorted[0] - 1e-9 or budget > rates_sorted[-1] + 1e-9:
        return None
    return float(np.interp(budget, rates_sorted, wers_sorted))


def _area_vs_random(curve: dict, random_curve: dict) -> float:
    """Area between trigger curve and random baseline.

    Positive = trigger better than random (lower WER at same escalation rate).
    """
    grid = np.linspace(0, 1, 101)

    def interp_wer(c: dict) -> np.ndarray:
        rates = np.array(c["escalation_rates"])
        wers = np.array(c["net_wers"])
        order = np.argsort(rates)
        return np.interp(grid, rates[order], wers[order])

    trigger_wers = interp_wer(curve)
    random_wers = interp_wer(random_curve)
    return float(np.trapezoid(random_wers - trigger_wers, grid))


def summarize(
    curve: dict,
    random_curve: dict,
    budget_points: list[float] | None = None,
) -> dict:
    """Compute summary scalars from an operating curve.

    Returns dict with netWER@{10,20,30,50}% and area_vs_random.
    """
    if budget_points is None:
        budget_points = [0.1, 0.2, 0.3, 0.5]

    result: dict = {}
    for bp in budget_points:
        pct = int(bp * 100)
        result[f"net_wer_at_{pct}pct"] = _interp_net_wer(curve, bp)

    result["area_vs_random"] = _area_vs_random(curve, random_curve)
    return result


# ---------------------------------------------------------------------------
# Decision scorecard
# ---------------------------------------------------------------------------

def decision_scorecard(
    scores: dict[str, float],
    gains: dict[str, float],
    tau: float,
) -> dict:
    """Binary classification metrics: does score predict gain > tau?

    Args:
        scores: utterance_id -> trigger score.
        gains: utterance_id -> escalation_gain (capped).
        tau: threshold for positive class (gain > tau).

    Returns:
        dict with auc, ap, spearman_r, pearson_r.
    """
    utt_ids = sorted(scores.keys())
    y_score = np.array([scores[uid] for uid in utt_ids])
    y_gain = np.array([gains[uid] for uid in utt_ids])
    y_binary = (y_gain > tau).astype(int)

    result: dict = {}

    # AUC — requires both classes present
    if len(np.unique(y_binary)) >= 2:
        result["auc"] = float(roc_auc_score(y_binary, y_score))
        result["ap"] = float(average_precision_score(y_binary, y_score))
    else:
        result["auc"] = None
        result["ap"] = None

    # Rank correlations on continuous gain
    if np.std(y_score) > 1e-8 and np.std(y_gain) > 1e-8:
        result["spearman_r"] = float(spearmanr(y_score, y_gain).statistic)
        result["pearson_r"] = float(pearsonr(y_score, y_gain).statistic)
    else:
        result["spearman_r"] = 0.0
        result["pearson_r"] = 0.0

    result["tau"] = tau
    result["n_positive"] = int(y_binary.sum())
    result["n_total"] = len(utt_ids)

    return result


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(
    scores: dict[str, float],
    wer_default: dict[str, float],
    wer_careful: dict[str, float],
    n: int = 1000,
    seed: int = 42,
    budget_points: list[float] | None = None,
    random_scores: dict[str, float] | None = None,
) -> dict:
    """Bootstrap CIs on summary scalars and curve bands.

    Resamples utterance IDs (preserving score/wer tuples), recomputes
    full curve per resample.

    Args:
        scores: trigger scores.
        wer_default: default model WERs (uncapped).
        wer_careful: careful model WERs (uncapped).
        n: number of bootstrap resamples.
        seed: random seed.
        budget_points: escalation rate budget points for summary.
        random_scores: random trigger scores (for area_vs_random).

    Returns:
        dict with point estimates, 95% CIs, and per-threshold curve bands.
    """
    if budget_points is None:
        budget_points = [0.1, 0.2, 0.3, 0.5]

    utt_ids = sorted(scores.keys())
    rng = np.random.RandomState(seed)

    # Point estimate
    point_curve = operating_curve(scores, wer_default, wer_careful)
    if random_scores is not None:
        random_curve = operating_curve(random_scores, wer_default, wer_careful)
    else:
        random_curve = point_curve  # fallback; area_vs_random will be 0
    point_summary = summarize(point_curve, random_curve, budget_points)

    # Bootstrap resamples
    boot_summaries: dict[str, list[float]] = defaultdict(list)
    boot_wer_curves: list[list[float]] = []

    for _ in range(n):
        idx = rng.choice(len(utt_ids), size=len(utt_ids), replace=True)
        sample_ids = [utt_ids[i] for i in idx]

        # Build resampled dicts (duplicates will just overwrite with same value)
        sample_scores = {uid: scores[uid] for uid in sample_ids}
        sample_default = {uid: wer_default[uid] for uid in sample_ids}
        sample_careful = {uid: wer_careful[uid] for uid in sample_ids}

        # But we need to handle duplicates properly — use indexed IDs
        indexed_scores = {}
        indexed_default = {}
        indexed_careful = {}
        for j, uid in enumerate(sample_ids):
            key = f"{uid}__{j}"
            indexed_scores[key] = scores[uid]
            indexed_default[key] = wer_default[uid]
            indexed_careful[key] = wer_careful[uid]

        b_curve = operating_curve(indexed_scores, indexed_default, indexed_careful)

        if random_scores is not None:
            indexed_random = {}
            for j, uid in enumerate(sample_ids):
                key = f"{uid}__{j}"
                indexed_random[key] = random_scores[uid]
            b_random = operating_curve(indexed_random, indexed_default, indexed_careful)
        else:
            b_random = b_curve

        b_summary = summarize(b_curve, b_random, budget_points)

        for k, v in b_summary.items():
            if v is not None:
                boot_summaries[k].append(v)

        boot_wer_curves.append(b_curve["net_wers"])

    # Compute CIs
    ci: dict = {}
    for k, vals in boot_summaries.items():
        arr = np.array(vals)
        ci[k] = {
            "point": point_summary.get(k),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "std": float(np.std(arr)),
        }

    # Curve bands
    wer_matrix = np.array(boot_wer_curves)  # (n, num_thresholds)
    curve_bands = {
        "thresholds": point_curve["thresholds"],
        "escalation_rates": point_curve["escalation_rates"],
        "net_wers_point": point_curve["net_wers"],
        "net_wers_ci_lo": np.percentile(wer_matrix, 2.5, axis=0).tolist(),
        "net_wers_ci_hi": np.percentile(wer_matrix, 97.5, axis=0).tolist(),
    }

    return {
        "point_summary": point_summary,
        "ci": ci,
        "curve_bands": curve_bands,
        "n_resamples": n,
    }


def paired_bootstrap(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    wer_default: dict[str, float],
    wer_careful: dict[str, float],
    n: int = 1000,
    seed: int = 42,
    budget_points: list[float] | None = None,
    random_scores: dict[str, float] | None = None,
) -> dict:
    """Bootstrap CI on difference (A - B) at each budget point.

    CI excluding 0 = statistically significant difference.

    Args:
        scores_a: trigger A scores.
        scores_b: trigger B scores.
        wer_default: default model WERs.
        wer_careful: careful model WERs.
        n: number of resamples.
        seed: random seed.
        budget_points: escalation rate budget points.
        random_scores: random trigger scores for area_vs_random.

    Returns:
        dict with per-metric CIs on (A - B) differences and significance flags.
    """
    if budget_points is None:
        budget_points = [0.1, 0.2, 0.3, 0.5]

    # Use only utterances present in both
    common_ids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    rng = np.random.RandomState(seed)

    diffs: dict[str, list[float]] = defaultdict(list)

    for _ in range(n):
        idx = rng.choice(len(common_ids), size=len(common_ids), replace=True)
        sample_ids = [common_ids[i] for i in idx]

        # Index to handle duplicates
        indexed_a, indexed_b = {}, {}
        indexed_default, indexed_careful = {}, {}
        indexed_random = {}
        for j, uid in enumerate(sample_ids):
            key = f"{uid}__{j}"
            indexed_a[key] = scores_a[uid]
            indexed_b[key] = scores_b[uid]
            indexed_default[key] = wer_default[uid]
            indexed_careful[key] = wer_careful[uid]
            if random_scores is not None:
                indexed_random[key] = random_scores[uid]

        curve_a = operating_curve(indexed_a, indexed_default, indexed_careful)
        curve_b = operating_curve(indexed_b, indexed_default, indexed_careful)

        if random_scores is not None:
            r_curve = operating_curve(indexed_random, indexed_default, indexed_careful)
        else:
            r_curve = curve_a  # fallback

        sum_a = summarize(curve_a, r_curve, budget_points)
        sum_b = summarize(curve_b, r_curve, budget_points)

        for k in sum_a:
            if sum_a[k] is not None and sum_b[k] is not None:
                diffs[k].append(sum_a[k] - sum_b[k])

    result: dict = {}
    for k, vals in diffs.items():
        arr = np.array(vals)
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        result[k] = {
            "mean_diff": float(np.mean(arr)),
            "ci_lo": lo,
            "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0),  # CI excludes 0
            "a_better": bool(lo > 0) if "area" in k else bool(hi < 0),
            # For net_wer: lower is better, so A better if diff < 0
            # For area_vs_random: higher is better, so A better if diff > 0
        }

    return {
        "diffs": result,
        "n_resamples": n,
        "n_utterances": len(common_ids),
    }


# ---------------------------------------------------------------------------
# Round 3: headroom summary
# ---------------------------------------------------------------------------

def headroom_summary(
    wer_default: dict[str, float],
    wer_careful: dict[str, float],
    scores_oracle: dict[str, float],
    scores_random: dict[str, float],
) -> dict:
    """Compute headroom diagnostics for a model pairing.

    Args:
        wer_default: utterance_id -> default model WER (uncapped).
        wer_careful: utterance_id -> careful model WER (uncapped).
        scores_oracle: utterance_id -> oracle trigger scores.
        scores_random: utterance_id -> random trigger scores.

    Returns:
        dict with oracle_area_vs_random, wer_gap, n_gain_positive,
        n_gain_above_005, median_gain, mean_gain.
    """
    utt_ids = sorted(set(wer_default.keys()) & set(wer_careful.keys()))
    gains = [escalation_gain(wer_default[uid], wer_careful[uid]) for uid in utt_ids]
    gains_arr = np.array(gains)

    capped_d = [cap_wer(wer_default[uid]) for uid in utt_ids]
    capped_c = [cap_wer(wer_careful[uid]) for uid in utt_ids]

    oracle_curve = operating_curve(scores_oracle, wer_default, wer_careful)
    random_curve = operating_curve(scores_random, wer_default, wer_careful)
    oracle_avr = _area_vs_random(oracle_curve, random_curve)

    return {
        "oracle_area_vs_random": oracle_avr,
        "wer_gap": float(np.mean(capped_d) - np.mean(capped_c)),
        "n_gain_positive": int(np.sum(gains_arr > 0)),
        "n_gain_above_005": int(np.sum(gains_arr > 0.05)),
        "median_gain": float(np.median(gains_arr)),
        "mean_gain": float(np.mean(gains_arr)),
    }


# ---------------------------------------------------------------------------
# Round 3: grid evaluation helper
# ---------------------------------------------------------------------------

def grid_eval(
    pairings: list[tuple[str, str]],
    triggers_fn: Callable[[tuple[str, str]], dict[str, dict[str, float]]],
    wer_loader: Callable[[str], dict[str, float]],
    random_seed: int = 42,
    budget_points: list[float] | None = None,
    bootstrap_n: int = 1000,
) -> dict[str, dict]:
    """Run full eval_common pipeline across a grid of model pairings.

    Args:
        pairings: list of (default_model, careful_model) tuples.
        triggers_fn: callable that takes a pairing tuple and returns
            dict of {trigger_name: {utterance_id: score}}.
        wer_loader: callable that takes a model path and returns
            {utterance_id: wer} dict from cache.
        random_seed: seed for random trigger and bootstrap.
        budget_points: escalation rate budget points.
        bootstrap_n: number of bootstrap resamples.

    Returns:
        dict keyed by "default__careful" pairing slug, each containing
        full eval results (triggers, headroom_summary, paired_bootstrap).
    """
    from accentedness_routing.triggers.random_trigger import RandomTrigger

    if budget_points is None:
        budget_points = [0.1, 0.2, 0.3, 0.5]

    results: dict[str, dict] = {}

    for default_model, careful_model in pairings:
        slug = f"{default_model.split('/')[-1]}__{careful_model.split('/')[-1]}"
        wer_default = wer_loader(default_model)
        wer_careful = wer_loader(careful_model)

        # Intersect utterance IDs
        common_ids = sorted(set(wer_default.keys()) & set(wer_careful.keys()))
        wd = {uid: wer_default[uid] for uid in common_ids}
        wc = {uid: wer_careful[uid] for uid in common_ids}

        # Random baseline
        random_trigger = RandomTrigger(seed=random_seed)
        random_scores = {uid: random_trigger.score(uid) for uid in common_ids}

        # Get triggers for this pairing
        all_triggers = triggers_fn((default_model, careful_model))
        all_triggers["random"] = random_scores

        gains = {uid: escalation_gain(wd[uid], wc[uid]) for uid in common_ids}

        pairing_results: dict[str, Any] = {"triggers": {}, "pairing": slug}

        confidence_scores = all_triggers.get("confidence")

        for trig_name, trig_scores in all_triggers.items():
            curve = operating_curve(trig_scores, wd, wc)
            random_curve = operating_curve(random_scores, wd, wc)
            summary = summarize(curve, random_curve, budget_points)

            scorecard_0 = decision_scorecard(trig_scores, gains, tau=0.0)
            scorecard_05 = decision_scorecard(trig_scores, gains, tau=0.05)

            boot = bootstrap(
                trig_scores, wd, wc,
                n=bootstrap_n, seed=random_seed,
                random_scores=random_scores,
                budget_points=budget_points,
            )

            entry: dict[str, Any] = {
                "summary": summary,
                "scorecard_tau_0.00": scorecard_0,
                "scorecard_tau_0.05": scorecard_05,
                "bootstrap_ci": boot["ci"],
                "curve_bands": boot["curve_bands"],
            }

            # Paired bootstrap vs confidence (if this isn't the confidence trigger)
            if confidence_scores is not None and trig_name != "confidence":
                paired = paired_bootstrap(
                    trig_scores, confidence_scores,
                    wd, wc,
                    n=bootstrap_n, seed=random_seed,
                    random_scores=random_scores,
                    budget_points=budget_points,
                )
                entry["paired_vs_confidence"] = paired

            pairing_results["triggers"][trig_name] = entry

        # Headroom summary
        oracle_scores = all_triggers.get("oracle", random_scores)
        pairing_results["headroom"] = headroom_summary(
            wd, wc, oracle_scores, random_scores
        )

        results[slug] = pairing_results

    return results


# ---------------------------------------------------------------------------
# Round 3: combiner evaluation
# ---------------------------------------------------------------------------

def combiner_eval(
    feature_matrix_val: np.ndarray,
    gains_val: np.ndarray,
    feature_matrix_test: np.ndarray,
    gains_test: np.ndarray,
    scores_confidence_test: dict[str, float],
    wer_default_test: dict[str, float],
    wer_careful_test: dict[str, float],
    random_scores_test: dict[str, float],
    tau: float = 0.05,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> dict:
    """Fit L2 logistic regression combiner on val, evaluate on test.

    Args:
        feature_matrix_val: (n_val, n_features) feature matrix for validation.
        gains_val: (n_val,) escalation gains for validation.
        feature_matrix_test: (n_test, n_features) feature matrix for test.
        gains_test: (n_test,) escalation gains for test.
        scores_confidence_test: utterance_id -> confidence score (for comparison).
        wer_default_test: utterance_id -> default WER.
        wer_careful_test: utterance_id -> careful WER.
        random_scores_test: utterance_id -> random scores.
        tau: gain threshold for binary target.
        bootstrap_n: number of bootstrap resamples.
        seed: random seed.

    Returns:
        dict with combiner_scores, val_auc, test_auc,
        paired_vs_confidence bootstrap result, and model info.
    """
    # Binary target: gain > tau
    y_val = (gains_val > tau).astype(int)
    y_test = (gains_test > tau).astype(int)

    # Fit L2 logistic regression with CV
    model = LogisticRegressionCV(
        Cs=10,
        l1_ratios=(0,),
        cv=5,
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )

    # Need both classes in training data
    if len(np.unique(y_val)) < 2:
        return {
            "error": "single_class_in_val",
            "n_positive_val": int(y_val.sum()),
            "n_total_val": len(y_val),
        }

    model.fit(feature_matrix_val, y_val)

    # Val AUC
    val_proba = model.predict_proba(feature_matrix_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_proba))

    # Test predictions
    test_proba = model.predict_proba(feature_matrix_test)[:, 1]

    # Test AUC
    if len(np.unique(y_test)) >= 2:
        test_auc = float(roc_auc_score(y_test, test_proba))
    else:
        test_auc = None

    # Build scores dict keyed by utterance_id (use test utterance order)
    test_utt_ids = sorted(wer_default_test.keys())
    combiner_scores = {uid: float(test_proba[i]) for i, uid in enumerate(test_utt_ids)}

    # Paired bootstrap vs confidence
    paired = paired_bootstrap(
        combiner_scores, scores_confidence_test,
        wer_default_test, wer_careful_test,
        n=bootstrap_n, seed=seed,
        random_scores=random_scores_test,
    )

    return {
        "combiner_scores": combiner_scores,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "best_C": float(model.C_[0]),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "paired_vs_confidence": paired,
        "family": "logistic",
    }
