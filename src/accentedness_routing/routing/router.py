"""Threshold sweep router: produces operating curves from trigger scores."""

from __future__ import annotations

import numpy as np

from accentedness_routing.triggers.base import RoutingTrigger


def compute_operating_curve(
    trigger: RoutingTrigger,
    utterance_ids: list[str],
    default_wers: dict[str, float],
    careful_wers: dict[str, float],
    num_thresholds: int = 101,
) -> dict:
    """Sweep thresholds to produce an operating curve.

    Returns dict with:
        - thresholds: list of threshold values
        - escalation_rates: fraction of utterances escalated at each threshold
        - net_wers: net WER at each threshold
        - trigger_name: name of the trigger
    """
    scores = {uid: trigger.score(uid) for uid in utterance_ids}
    thresholds = np.linspace(0, 1, num_thresholds).tolist()

    escalation_rates = []
    net_wers = []

    for thresh in thresholds:
        escalated = [uid for uid in utterance_ids if scores[uid] >= thresh]
        not_escalated = [uid for uid in utterance_ids if scores[uid] < thresh]

        esc_rate = len(escalated) / len(utterance_ids)

        # Net WER: careful model for escalated, default for rest
        wer_sum = 0.0
        for uid in escalated:
            wer_sum += careful_wers[uid]
        for uid in not_escalated:
            wer_sum += default_wers[uid]
        net_wer = wer_sum / len(utterance_ids)

        escalation_rates.append(esc_rate)
        net_wers.append(net_wer)

    return {
        "trigger_name": trigger.name,
        "thresholds": thresholds,
        "escalation_rates": escalation_rates,
        "net_wers": net_wers,
    }
