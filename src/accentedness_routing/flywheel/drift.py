"""Drift detection via KS test on scalar score distributions."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def simulate_accent_shift(
    scores_by_accent: dict[str, list[float]],
    original_mix: dict[str, float] | None = None,
    shifted_mix: dict[str, float] | None = None,
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    """Simulate distribution shift by changing accent mix and run KS test.

    Args:
        scores_by_accent: accent → list of scalar scores
        original_mix: accent → proportion (default: uniform)
        shifted_mix: accent → proportion (shifted toward hard accents)
        n_samples: number of samples to draw for each distribution

    Returns:
        Dict with KS statistic, p-value, and sample distributions.
    """
    rng = np.random.RandomState(seed)
    accents = sorted(scores_by_accent.keys())

    if original_mix is None:
        original_mix = {a: 1.0 / len(accents) for a in accents}
    if shifted_mix is None:
        # Shift toward accents with higher mean scores (harder)
        means = {a: np.mean(scores_by_accent[a]) for a in accents}
        total = sum(means.values())
        shifted_mix = {a: means[a] / total for a in accents}

    def sample_from_mix(mix, n):
        samples = []
        for accent in accents:
            k = max(1, int(n * mix.get(accent, 0)))
            pool = scores_by_accent[accent]
            if pool:
                drawn = rng.choice(pool, size=min(k, len(pool)), replace=True)
                samples.extend(drawn.tolist())
        return samples

    original_samples = sample_from_mix(original_mix, n_samples)
    shifted_samples = sample_from_mix(shifted_mix, n_samples)

    stat, pval = ks_2samp(original_samples, shifted_samples)

    return {
        "ks_statistic": float(stat),
        "p_value": float(pval),
        "detected": pval < 0.01,
        "n_original": len(original_samples),
        "n_shifted": len(shifted_samples),
        "original_mean": float(np.mean(original_samples)),
        "shifted_mean": float(np.mean(shifted_samples)),
    }
