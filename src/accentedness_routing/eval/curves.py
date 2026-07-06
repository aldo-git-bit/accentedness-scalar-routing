"""Operating curve computation and comparison."""

from __future__ import annotations

from accentedness_routing.routing.metrics import compute_summary, dominates
from accentedness_routing.routing.router import compute_operating_curve
from accentedness_routing.triggers.base import RoutingTrigger


def compute_all_curves(
    triggers: list[RoutingTrigger],
    utterance_ids: list[str],
    default_wers: dict[str, float],
    careful_wers: dict[str, float],
    num_thresholds: int = 101,
    budget_points: list[float] | None = None,
) -> dict:
    """Compute operating curves and summaries for all triggers."""
    curves = {}
    for trigger in triggers:
        curve = compute_operating_curve(
            trigger, utterance_ids, default_wers, careful_wers, num_thresholds
        )
        curves[trigger.name] = curve

    # Find random curve for reference
    random_curve = curves.get("random")
    if random_curve is None:
        # Create a dummy linear reference
        random_curve = curves[list(curves.keys())[0]]

    summaries = {}
    for name, curve in curves.items():
        summaries[name] = compute_summary(curve, random_curve, budget_points)

    # Dominance checks
    dominance = {}
    names = list(curves.keys())
    for a in names:
        for b in names:
            if a != b:
                dominance[f"{a}_dominates_{b}"] = dominates(curves[a], curves[b])

    return {
        "curves": curves,
        "summaries": summaries,
        "dominance": dominance,
    }
