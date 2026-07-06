"""Per-accent slicing analysis."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def per_accent_analysis(
    utterance_ids: list[str],
    accent_map: dict[str, str],
    default_wers: dict[str, float],
    careful_wers: dict[str, float],
    trigger_scores: dict[str, dict[str, float]],
) -> dict:
    """Compute per-accent WER and score statistics.

    Args:
        trigger_scores: trigger_name → {utterance_id → score}

    Returns:
        Dict with per-accent breakdowns.
    """
    # Group utterances by accent
    accent_groups: dict[str, list[str]] = defaultdict(list)
    for uid in utterance_ids:
        accent_groups[accent_map[uid]].append(uid)

    results = {}
    for accent, uids in sorted(accent_groups.items()):
        d_wers = [default_wers[uid] for uid in uids]
        c_wers = [careful_wers[uid] for uid in uids]
        gains = [default_wers[uid] - careful_wers[uid] for uid in uids]

        accent_result = {
            "n_utterances": len(uids),
            "mean_default_wer": float(np.mean(d_wers)),
            "mean_careful_wer": float(np.mean(c_wers)),
            "mean_escalation_gain": float(np.mean(gains)),
            "std_default_wer": float(np.std(d_wers)),
        }

        # Per-trigger score stats
        for trigger_name, scores in trigger_scores.items():
            t_scores = [scores[uid] for uid in uids if uid in scores]
            if t_scores:
                accent_result[f"{trigger_name}_mean_score"] = float(np.mean(t_scores))
                accent_result[f"{trigger_name}_std_score"] = float(np.std(t_scores))

        results[accent] = accent_result

    return results
