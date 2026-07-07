"""Unit tests for Round 3 eval_common additions: headroom_summary, combiner_eval."""

from __future__ import annotations

import numpy as np
import pytest

from accentedness_routing.eval.eval_common import (
    combiner_eval,
    headroom_summary,
)


# ---------------------------------------------------------------------------
# headroom_summary
# ---------------------------------------------------------------------------


class TestHeadroomSummary:
    @pytest.fixture()
    def data(self):
        """Synthetic data: 20 utterances, clear gap between default and careful."""
        rng = np.random.RandomState(42)
        n = 20
        utt_ids = [f"u{i}" for i in range(n)]
        # Default WER: beta(2, 3) => mean ~0.4
        wer_default = {uid: float(rng.beta(2, 3)) for uid in utt_ids}
        # Careful WER: default * uniform(0.2, 0.6) => clearly lower
        wer_careful = {uid: max(0.0, wer_default[uid] * rng.uniform(0.2, 0.6))
                       for uid in utt_ids}
        # Oracle: higher score for higher gain
        gains = {uid: max(0.0, min(1.0, wer_default[uid] - wer_careful[uid]))
                 for uid in utt_ids}
        scores_oracle = {uid: gains[uid] for uid in utt_ids}
        scores_random = {uid: float(rng.rand()) for uid in utt_ids}
        return wer_default, wer_careful, scores_oracle, scores_random

    def test_wer_gap_positive(self, data):
        """WER gap should be positive (default worse than careful)."""
        wer_default, wer_careful, scores_oracle, scores_random = data
        result = headroom_summary(wer_default, wer_careful, scores_oracle, scores_random)
        assert result["wer_gap"] > 0

    def test_oracle_area_positive(self, data):
        """Oracle area-vs-random should be positive."""
        wer_default, wer_careful, scores_oracle, scores_random = data
        result = headroom_summary(wer_default, wer_careful, scores_oracle, scores_random)
        assert result["oracle_area_vs_random"] > 0

    def test_gain_counts(self, data):
        """n_gain_positive should be <= total, n_gain_above_005 <= n_gain_positive."""
        wer_default, wer_careful, scores_oracle, scores_random = data
        result = headroom_summary(wer_default, wer_careful, scores_oracle, scores_random)
        n_total = len(wer_default)
        assert 0 <= result["n_gain_above_005"] <= result["n_gain_positive"] <= n_total

    def test_mean_median_gain(self, data):
        """Mean and median gain should be positive for this data."""
        wer_default, wer_careful, scores_oracle, scores_random = data
        result = headroom_summary(wer_default, wer_careful, scores_oracle, scores_random)
        assert result["mean_gain"] > 0
        assert result["median_gain"] > 0

    def test_no_gap_scenario(self):
        """When default == careful, gains are zero."""
        utt_ids = [f"u{i}" for i in range(10)]
        wer = {uid: 0.3 for uid in utt_ids}
        rng = np.random.RandomState(0)
        scores_random = {uid: float(rng.rand()) for uid in utt_ids}
        result = headroom_summary(wer, wer, wer, scores_random)
        assert result["wer_gap"] == pytest.approx(0.0)
        assert result["n_gain_positive"] == 0
        assert result["mean_gain"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# combiner_eval
# ---------------------------------------------------------------------------


class TestCombinerEval:
    def test_perfect_features_high_auc(self):
        """Perfect features should yield high AUC."""
        rng = np.random.RandomState(42)
        n_val, n_test = 60, 40

        # Create features that perfectly predict gain > 0.05
        val_gains = np.concatenate([np.zeros(30), np.ones(30) * 0.2])
        val_features = np.column_stack([val_gains, rng.randn(n_val, 2) * 0.01])

        test_gains = np.concatenate([np.zeros(20), np.ones(20) * 0.2])
        test_features = np.column_stack([test_gains, rng.randn(n_test, 2) * 0.01])

        test_utt_ids = [f"u{i}" for i in range(n_test)]
        # Dummy WER/confidence/random dicts
        wer_default = {uid: 0.5 for uid in test_utt_ids}
        wer_careful = {uid: 0.3 for uid in test_utt_ids}
        scores_confidence = {uid: float(rng.rand()) for uid in test_utt_ids}
        random_scores = {uid: float(rng.rand()) for uid in test_utt_ids}

        result = combiner_eval(
            val_features, val_gains,
            test_features, test_gains,
            scores_confidence, wer_default, wer_careful, random_scores,
            tau=0.05, bootstrap_n=50, seed=42,
        )

        assert "error" not in result
        assert result["val_auc"] > 0.9
        assert result["test_auc"] is not None and result["test_auc"] > 0.8

    def test_random_features_low_auc(self):
        """Random features should yield ~0.5 AUC."""
        rng = np.random.RandomState(42)
        n_val, n_test = 100, 50

        val_gains = rng.rand(n_val)
        val_features = rng.randn(n_val, 5)

        test_gains = rng.rand(n_test)
        test_features = rng.randn(n_test, 5)

        test_utt_ids = [f"u{i}" for i in range(n_test)]
        wer_default = {uid: 0.5 for uid in test_utt_ids}
        wer_careful = {uid: 0.3 for uid in test_utt_ids}
        scores_confidence = {uid: float(rng.rand()) for uid in test_utt_ids}
        random_scores = {uid: float(rng.rand()) for uid in test_utt_ids}

        result = combiner_eval(
            val_features, val_gains,
            test_features, test_gains,
            scores_confidence, wer_default, wer_careful, random_scores,
            tau=0.5, bootstrap_n=50, seed=42,
        )

        assert "error" not in result
        # AUC should be near 0.5 for random features
        assert 0.3 < result["val_auc"] < 0.75

    def test_single_class_error(self):
        """If val has only one class, return error dict."""
        rng = np.random.RandomState(42)
        n_val, n_test = 20, 10

        val_gains = np.zeros(n_val)  # all below tau
        val_features = rng.randn(n_val, 3)

        test_gains = rng.rand(n_test)
        test_features = rng.randn(n_test, 3)

        test_utt_ids = [f"u{i}" for i in range(n_test)]
        wer_default = {uid: 0.5 for uid in test_utt_ids}
        wer_careful = {uid: 0.3 for uid in test_utt_ids}
        scores_confidence = {uid: float(rng.rand()) for uid in test_utt_ids}
        random_scores = {uid: float(rng.rand()) for uid in test_utt_ids}

        result = combiner_eval(
            val_features, val_gains,
            test_features, test_gains,
            scores_confidence, wer_default, wer_careful, random_scores,
            tau=0.05, bootstrap_n=50, seed=42,
        )

        assert result["error"] == "single_class_in_val"

    def test_returns_model_info(self):
        """Result should include coef, intercept, best_C."""
        rng = np.random.RandomState(42)
        n_val, n_test = 60, 30

        val_gains = np.concatenate([np.zeros(30), np.ones(30) * 0.2])
        val_features = rng.randn(n_val, 4)

        test_gains = np.concatenate([np.zeros(15), np.ones(15) * 0.2])
        test_features = rng.randn(n_test, 4)

        test_utt_ids = [f"u{i}" for i in range(n_test)]
        wer_default = {uid: 0.5 for uid in test_utt_ids}
        wer_careful = {uid: 0.3 for uid in test_utt_ids}
        scores_confidence = {uid: float(rng.rand()) for uid in test_utt_ids}
        random_scores = {uid: float(rng.rand()) for uid in test_utt_ids}

        result = combiner_eval(
            val_features, val_gains,
            test_features, test_gains,
            scores_confidence, wer_default, wer_careful, random_scores,
            tau=0.05, bootstrap_n=50, seed=42,
        )

        assert "coef" in result
        assert "intercept" in result
        assert "best_C" in result
        assert result["family"] == "logistic"
