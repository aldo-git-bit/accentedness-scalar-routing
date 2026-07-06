"""Summary metrics computed from operating curves."""

from __future__ import annotations

import numpy as np


def net_wer_at_budget(curve: dict, budget: float) -> float | None:
    """Interpolate net WER at a given escalation budget.

    Returns None if the budget is outside the curve range.
    """
    rates = curve["escalation_rates"]
    wers = curve["net_wers"]

    # Find the two points bracketing the budget
    for i in range(len(rates) - 1):
        if rates[i] <= budget <= rates[i + 1]:
            # Linear interpolation
            if rates[i + 1] == rates[i]:
                return wers[i]
            t = (budget - rates[i]) / (rates[i + 1] - rates[i])
            return wers[i] + t * (wers[i + 1] - wers[i])
        elif rates[i] >= budget >= rates[i + 1]:
            # Rates might be decreasing (threshold going up)
            if rates[i] == rates[i + 1]:
                return wers[i]
            t = (budget - rates[i + 1]) / (rates[i] - rates[i + 1])
            return wers[i + 1] + t * (wers[i] - wers[i + 1])

    # Budget at boundary
    if abs(rates[0] - budget) < 1e-6:
        return wers[0]
    if abs(rates[-1] - budget) < 1e-6:
        return wers[-1]

    return None


def area_vs_random(curve: dict, random_curve: dict) -> float:
    """Area between the trigger's curve and random baseline.

    Positive = trigger is better than random (lower WER at same escalation rate).
    """
    # Interpolate both curves onto a common grid
    grid = np.linspace(0, 1, 101)

    def interp_wer(c, grid):
        rates = np.array(c["escalation_rates"])
        wers = np.array(c["net_wers"])
        # Sort by escalation rate for interpolation
        order = np.argsort(rates)
        return np.interp(grid, rates[order], wers[order])

    trigger_wers = interp_wer(curve, grid)
    random_wers = interp_wer(random_curve, grid)

    # Area where random is above trigger (trigger is better)
    return float(np.trapezoid(random_wers - trigger_wers, grid))


def dominates(curve_a: dict, curve_b: dict) -> bool:
    """Check if curve A dominates curve B (lower WER at all escalation rates)."""
    grid = np.linspace(0, 1, 101)

    def interp_wer(c, grid):
        rates = np.array(c["escalation_rates"])
        wers = np.array(c["net_wers"])
        order = np.argsort(rates)
        return np.interp(grid, rates[order], wers[order])

    a_wers = interp_wer(curve_a, grid)
    b_wers = interp_wer(curve_b, grid)

    return bool(np.all(a_wers <= b_wers + 1e-6))


def compute_summary(
    curve: dict,
    random_curve: dict,
    budget_points: list[float] | None = None,
) -> dict:
    """Compute all summary metrics for a curve."""
    if budget_points is None:
        budget_points = [0.1, 0.2, 0.3, 0.5]

    summary = {"trigger_name": curve["trigger_name"]}

    for bp in budget_points:
        wer = net_wer_at_budget(curve, bp)
        summary[f"net_wer_at_{int(bp*100)}pct"] = wer

    summary["area_vs_random"] = area_vs_random(curve, random_curve)

    return summary
