"""Hard-case mining: find utterances with highest scalar scores or misrouting cost."""

from __future__ import annotations


def mine_hard_cases(
    utterance_ids: list[str],
    scores: dict[str, float],
    default_wers: dict[str, float],
    careful_wers: dict[str, float],
    top_k: int = 20,
) -> dict:
    """Find the most informative utterances for retraining.

    Returns:
        - highest_score: top-k utterances with highest escalation scores
        - highest_misrouting_cost: top-k where not escalating costs the most WER
    """
    # Highest scalar scores
    by_score = sorted(utterance_ids, key=lambda uid: scores.get(uid, 0), reverse=True)

    # Highest misrouting cost = WER_default - WER_careful (what you lose by not escalating)
    costs = {uid: default_wers[uid] - careful_wers[uid] for uid in utterance_ids}
    by_cost = sorted(utterance_ids, key=lambda uid: costs[uid], reverse=True)

    return {
        "highest_score": [
            {
                "utterance_id": uid,
                "score": scores.get(uid, 0),
                "default_wer": default_wers[uid],
                "careful_wer": careful_wers[uid],
                "cost": costs[uid],
            }
            for uid in by_score[:top_k]
        ],
        "highest_misrouting_cost": [
            {
                "utterance_id": uid,
                "score": scores.get(uid, 0),
                "default_wer": default_wers[uid],
                "careful_wer": careful_wers[uid],
                "cost": costs[uid],
            }
            for uid in by_cost[:top_k]
        ],
    }
