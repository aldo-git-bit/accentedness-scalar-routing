"""Diagnostic analyses: speaker leakage, error profiles."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import mutual_info_score


def speaker_leakage_check(
    utterance_ids: list[str],
    scores: dict[str, float],
    speaker_map: dict[str, str],
    accent_map: dict[str, str],
) -> dict:
    """Check if scalar scores leak speaker identity vs just accent.

    Computes mutual information between:
    - scores and speakers (should be low if not leaking)
    - scores and accents (expected to be higher)
    """
    score_vals = [scores[uid] for uid in utterance_ids]
    speakers = [speaker_map[uid] for uid in utterance_ids]
    accents = [accent_map[uid] for uid in utterance_ids]

    # Discretize scores into bins for MI computation
    score_bins = np.digitize(score_vals, bins=np.linspace(0, 1, 11)).tolist()

    mi_speaker = mutual_info_score(score_bins, speakers)
    mi_accent = mutual_info_score(score_bins, accents)

    return {
        "mi_score_speaker": float(mi_speaker),
        "mi_score_accent": float(mi_accent),
        "ratio_speaker_to_accent": float(mi_speaker / mi_accent) if mi_accent > 0 else float("inf"),
    }


def error_profile(
    utterance_ids: list[str],
    default_wers: dict[str, float],
    careful_wers: dict[str, float],
    accent_map: dict[str, str],
) -> dict:
    """Compute error profiles per accent (which accents benefit most from escalation)."""
    accent_groups: dict[str, list[str]] = defaultdict(list)
    for uid in utterance_ids:
        accent_groups[accent_map[uid]].append(uid)

    profiles = {}
    for accent, uids in sorted(accent_groups.items()):
        gains = [default_wers[uid] - careful_wers[uid] for uid in uids]
        worse_with_careful = sum(1 for g in gains if g < 0)

        profiles[accent] = {
            "mean_gain": float(np.mean(gains)),
            "median_gain": float(np.median(gains)),
            "max_gain": float(np.max(gains)),
            "pct_worse_with_careful": worse_with_careful / len(uids) * 100,
            "n_utterances": len(uids),
        }

    return profiles
