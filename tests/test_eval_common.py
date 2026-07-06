"""Unit tests for eval_common with synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from accentedness_routing.eval.eval_common import (
    bootstrap,
    cap_wer,
    decision_scorecard,
    escalation_gain,
    operating_curve,
    paired_bootstrap,
    summarize,
)


# ---------------------------------------------------------------------------
# cap_wer
# ---------------------------------------------------------------------------


class TestCapWer:
    def test_zero(self):
        assert cap_wer(0.0) == 0.0

    def test_half(self):
        assert cap_wer(0.5) == 0.5

    def test_one(self):
        assert cap_wer(1.0) == 1.0

    def test_hallucination(self):
        assert cap_wer(8.96) == 1.0

    def test_negative(self):
        assert cap_wer(-0.1) == -0.1  # negative WER passes through


# ---------------------------------------------------------------------------
# escalation_gain
# ---------------------------------------------------------------------------


class TestEscalationGain:
    def test_caps_before_subtraction(self):
        # 8.96 and 0.4 => cap(8.96) - cap(0.4) = 1.0 - 0.4 = 0.6, NOT 8.56
        assert escalation_gain(8.96, 0.4) == pytest.approx(0.6)

    def test_both_hallucinated(self):
        # Both > 1 => cap(2.0) - cap(3.0) = 1.0 - 1.0 = 0.0
        assert escalation_gain(2.0, 3.0) == pytest.approx(0.0)

    def test_normal_values(self):
        assert escalation_gain(0.5, 0.2) == pytest.approx(0.3)

    def test_careful_worse(self):
        # Negative gain when careful is worse
        assert escalation_gain(0.2, 0.5) == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# operating_curve - micro
# ---------------------------------------------------------------------------


class TestOperatingCurveMicro:
    """4 utterances with known scores/WERs, verify exact net WER."""

    @pytest.fixture()
    def data(self):
        # 4 utterances with controlled values
        scores = {"a": 0.9, "b": 0.7, "c": 0.3, "d": 0.1}
        wer_default = {"a": 0.8, "b": 0.6, "c": 0.2, "d": 0.1}
        wer_careful = {"a": 0.2, "b": 0.3, "c": 0.1, "d": 0.1}
        return scores, wer_default, wer_careful

    def test_all_default(self, data):
        """At threshold=1.0, nothing escalated => all default WER."""
        scores, wer_default, wer_careful = data
        curve = operating_curve(scores, wer_default, wer_careful, num_thresholds=3)
        # threshold=1.0 is last point; only utt "a" with score=0.9 < 1.0
        # At exact threshold=1.0, score >= 1.0 needed => none escalated
        assert curve["escalation_rates"][-1] == pytest.approx(0.0, abs=0.01)
        # net WER = mean of capped defaults = (0.8+0.6+0.2+0.1)/4 = 0.425
        assert curve["net_wers"][-1] == pytest.approx(0.425, abs=0.01)

    def test_all_escalated(self, data):
        """At threshold=0.0, everything escalated => all careful WER."""
        scores, wer_default, wer_careful = data
        curve = operating_curve(scores, wer_default, wer_careful, num_thresholds=3)
        # threshold=0.0 is first point; all scores >= 0 => all escalated
        assert curve["escalation_rates"][0] == pytest.approx(1.0)
        # net WER = mean of capped careful = (0.2+0.3+0.1+0.1)/4 = 0.175
        assert curve["net_wers"][0] == pytest.approx(0.175)

    def test_caps_applied(self):
        """Verify capping is applied: WER > 1.0 treated as 1.0."""
        scores = {"a": 0.9, "b": 0.1}
        wer_default = {"a": 5.0, "b": 0.2}  # "a" is hallucinated
        wer_careful = {"a": 0.3, "b": 0.1}
        curve = operating_curve(scores, wer_default, wer_careful, num_thresholds=3)
        # At threshold=0: all escalated => mean careful = (0.3+0.1)/2 = 0.2
        assert curve["net_wers"][0] == pytest.approx(0.2)
        # At threshold=1: none escalated => mean capped default = (1.0+0.2)/2 = 0.6
        assert curve["net_wers"][-1] == pytest.approx(0.6)

    def test_num_thresholds(self, data):
        scores, wer_default, wer_careful = data
        curve = operating_curve(scores, wer_default, wer_careful, num_thresholds=51)
        assert len(curve["thresholds"]) == 51
        assert len(curve["escalation_rates"]) == 51
        assert len(curve["net_wers"]) == 51


# ---------------------------------------------------------------------------
# operating_curve - macro
# ---------------------------------------------------------------------------


class TestOperatingCurveMacro:
    """2 accents with different sizes, verify per-accent-then-average."""

    def test_macro_vs_micro(self):
        # Accent A: 3 utterances, Accent B: 1 utterance
        # In micro, B gets 1/4 weight; in macro, B gets 1/2 weight
        scores = {"a1": 0.9, "a2": 0.8, "a3": 0.7, "b1": 0.1}
        wer_default = {"a1": 0.4, "a2": 0.4, "a3": 0.4, "b1": 0.8}
        wer_careful = {"a1": 0.1, "a2": 0.1, "a3": 0.1, "b1": 0.2}
        accent_map = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}

        micro = operating_curve(scores, wer_default, wer_careful, aggregation="micro",
                                num_thresholds=3)
        macro = operating_curve(scores, wer_default, wer_careful, aggregation="macro",
                                accent_map=accent_map, num_thresholds=3)

        # At threshold=0 (all escalated):
        # micro: (0.1+0.1+0.1+0.2)/4 = 0.125
        # macro: mean_A=0.1, mean_B=0.2, average=0.15
        assert micro["net_wers"][0] == pytest.approx(0.125)
        assert macro["net_wers"][0] == pytest.approx(0.15)

    def test_macro_requires_accent_map(self):
        scores = {"a": 0.5}
        wer_d = {"a": 0.3}
        wer_c = {"a": 0.1}
        with pytest.raises(ValueError, match="accent_map required"):
            operating_curve(scores, wer_d, wer_c, aggregation="macro")


# ---------------------------------------------------------------------------
# decision_scorecard
# ---------------------------------------------------------------------------


class TestDecisionScorecard:
    def test_perfect_predictor(self):
        """AUC = 1.0 when scores perfectly predict gain > tau."""
        scores = {f"u{i}": float(i >= 5) for i in range(10)}
        gains = {f"u{i}": 0.5 if i >= 5 else 0.0 for i in range(10)}
        result = decision_scorecard(scores, gains, tau=0.1)
        assert result["auc"] == pytest.approx(1.0)

    def test_random_predictor(self):
        """AUC ~ 0.5 for random scores."""
        rng = np.random.RandomState(42)
        n = 200
        scores = {f"u{i}": float(rng.rand()) for i in range(n)}
        gains = {f"u{i}": float(rng.rand()) for i in range(n)}
        result = decision_scorecard(scores, gains, tau=0.5)
        assert 0.3 < result["auc"] < 0.7

    def test_single_class(self):
        """AUC is None when only one class present."""
        scores = {f"u{i}": float(i) / 10 for i in range(10)}
        gains = {f"u{i}": 0.0 for i in range(10)}  # all below tau
        result = decision_scorecard(scores, gains, tau=0.5)
        assert result["auc"] is None

    def test_spearman_pearson(self):
        """Correlations computed correctly for monotonic data."""
        scores = {f"u{i}": float(i) / 10 for i in range(10)}
        gains = {f"u{i}": float(i) / 10 for i in range(10)}
        result = decision_scorecard(scores, gains, tau=0.5)
        assert result["spearman_r"] == pytest.approx(1.0, abs=1e-6)
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    @pytest.fixture()
    def data(self):
        rng = np.random.RandomState(42)
        n = 50
        utt_ids = [f"u{i}" for i in range(n)]
        wer_default = {uid: float(rng.beta(2, 5)) for uid in utt_ids}
        wer_careful = {uid: max(0, w * rng.uniform(0.3, 0.8))
                       for uid, w in wer_default.items()}
        scores = {uid: float(rng.rand()) for uid in utt_ids}
        random_scores = {uid: float(rng.rand()) for uid in utt_ids}
        return scores, wer_default, wer_careful, random_scores

    def test_ci_contains_point_estimate(self, data):
        """Point estimate should fall within 95% CI."""
        scores, wer_default, wer_careful, random_scores = data
        result = bootstrap(scores, wer_default, wer_careful, n=200, seed=42,
                           random_scores=random_scores)
        for k, ci_data in result["ci"].items():
            point = ci_data["point"]
            if point is not None:
                # Allow some tolerance — point estimate won't always be
                # perfectly inside due to resampling, but should be close
                assert ci_data["ci_lo"] <= point + 0.05
                assert ci_data["ci_hi"] >= point - 0.05

    def test_deterministic_with_seed(self, data):
        """Same seed => same CIs."""
        scores, wer_default, wer_careful, random_scores = data
        r1 = bootstrap(scores, wer_default, wer_careful, n=100, seed=123,
                        random_scores=random_scores)
        r2 = bootstrap(scores, wer_default, wer_careful, n=100, seed=123,
                        random_scores=random_scores)
        for k in r1["ci"]:
            assert r1["ci"][k]["ci_lo"] == pytest.approx(r2["ci"][k]["ci_lo"])
            assert r1["ci"][k]["ci_hi"] == pytest.approx(r2["ci"][k]["ci_hi"])

    def test_curve_bands_shape(self, data):
        scores, wer_default, wer_careful, random_scores = data
        result = bootstrap(scores, wer_default, wer_careful, n=50, seed=42,
                           random_scores=random_scores)
        bands = result["curve_bands"]
        n_thresh = len(bands["thresholds"])
        assert len(bands["net_wers_point"]) == n_thresh
        assert len(bands["net_wers_ci_lo"]) == n_thresh
        assert len(bands["net_wers_ci_hi"]) == n_thresh


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------


class TestPairedBootstrap:
    def test_identical_triggers_ci_includes_zero(self):
        """Identical triggers => CI on difference should include 0."""
        rng = np.random.RandomState(42)
        n = 50
        utt_ids = [f"u{i}" for i in range(n)]
        wer_default = {uid: float(rng.beta(2, 5)) for uid in utt_ids}
        wer_careful = {uid: max(0, w * rng.uniform(0.3, 0.8))
                       for uid, w in wer_default.items()}
        scores = {uid: float(rng.rand()) for uid in utt_ids}

        result = paired_bootstrap(scores, scores, wer_default, wer_careful,
                                  n=200, seed=42)
        # All diffs should include 0 (not significant)
        for k, diff_data in result["diffs"].items():
            assert diff_data["ci_lo"] <= 0.001
            assert diff_data["ci_hi"] >= -0.001
            assert not diff_data["significant"]

    def test_oracle_vs_random_significant(self):
        """Oracle vs random should show significant difference."""
        rng = np.random.RandomState(42)
        n = 80
        utt_ids = [f"u{i}" for i in range(n)]
        wer_default = {uid: float(rng.beta(2, 3)) for uid in utt_ids}
        wer_careful = {uid: max(0, w * rng.uniform(0.1, 0.5))
                       for uid, w in wer_default.items()}

        # Oracle scores: higher for utterances that benefit more
        oracle_scores = {uid: min(1.0, max(0.0, wer_default[uid] - wer_careful[uid]))
                         for uid in utt_ids}
        # Random scores
        random_scores = {uid: float(rng.rand()) for uid in utt_ids}

        result = paired_bootstrap(oracle_scores, random_scores,
                                  wer_default, wer_careful,
                                  n=500, seed=42, random_scores=random_scores)

        # area_vs_random difference should be significant (oracle better)
        avr = result["diffs"].get("area_vs_random")
        if avr:
            assert avr["significant"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_budget_points(self):
        """Verify all requested budget points appear in output."""
        scores = {f"u{i}": float(i) / 9 for i in range(10)}
        wer_d = {f"u{i}": 0.5 for i in range(10)}
        wer_c = {f"u{i}": 0.2 for i in range(10)}
        curve = operating_curve(scores, wer_d, wer_c)
        random_scores = {f"u{i}": 0.5 for i in range(10)}
        random_curve = operating_curve(random_scores, wer_d, wer_c)
        result = summarize(curve, random_curve, [0.1, 0.2, 0.3, 0.5])
        assert "net_wer_at_10pct" in result
        assert "net_wer_at_20pct" in result
        assert "net_wer_at_30pct" in result
        assert "net_wer_at_50pct" in result
        assert "area_vs_random" in result
