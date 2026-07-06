"""Tiny end-to-end smoke test: validates full pipeline on synthetic data."""

import numpy as np
import torch

from accentedness_routing.asr.wer import compute_wer
from accentedness_routing.features.pooling import LearnableWeightedSum
from accentedness_routing.routing.metrics import compute_summary, net_wer_at_budget
from accentedness_routing.routing.router import compute_operating_curve
from accentedness_routing.triggers.oracle import OracleTrigger
from accentedness_routing.triggers.random_trigger import RandomTrigger
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger
from accentedness_routing.triggers.train_probe import train_probe


def test_e2e_tiny():
    """Run a tiny version of the full pipeline with synthetic data."""
    n = 20
    num_layers = 25
    hidden_dim = 1024

    # Simulate utterance IDs, WERs, and features
    utt_ids = [f"utt_{i:03d}" for i in range(n)]
    rng = np.random.RandomState(42)

    # Simulate WERs where some utterances are "hard"
    default_wers = {uid: float(rng.beta(2, 5)) for uid in utt_ids}
    # Careful model reduces WER for hard utterances
    careful_wers = {uid: max(0, w * rng.uniform(0.3, 0.8)) for uid, w in default_wers.items()}

    # Simulate WavLM features — make them correlate with difficulty
    features = {}
    for uid in utt_ids:
        base = torch.randn(num_layers, hidden_dim)
        # Add difficulty signal to later layers
        difficulty = default_wers[uid]
        base[20:] += difficulty * 0.5
        features[uid] = base

    # 1. Test WER computation
    wer = compute_wer("the cat sat on the mat", "the cat on the mat")
    assert 0 < wer < 1

    # 2. Test triggers
    oracle = OracleTrigger(default_wers, careful_wers)
    random = RandomTrigger(seed=42)

    oracle_scores = [oracle.score(uid) for uid in utt_ids]
    assert all(0 <= s <= 1 for s in oracle_scores)

    # 3. Test operating curve
    curve = compute_operating_curve(oracle, utt_ids, default_wers, careful_wers, num_thresholds=11)
    assert len(curve["escalation_rates"]) == 11
    assert len(curve["net_wers"]) == 11

    random_curve = compute_operating_curve(random, utt_ids, default_wers, careful_wers, num_thresholds=11)

    # 4. Test metrics
    wer_20 = net_wer_at_budget(curve, 0.2)
    assert wer_20 is not None
    summary = compute_summary(curve, random_curve, [0.2, 0.5])
    assert "net_wer_at_20pct" in summary

    # 5. Test probe training (tiny)
    train_X = torch.stack([features[uid] for uid in utt_ids[:14]])
    train_y = torch.tensor([default_wers[uid] for uid in utt_ids[:14]], dtype=torch.float32)
    val_X = torch.stack([features[uid] for uid in utt_ids[14:]])
    val_y = torch.tensor([default_wers[uid] for uid in utt_ids[14:]], dtype=torch.float32)

    model, history = train_probe(
        train_X, train_y, val_X, val_y,
        num_layers=num_layers, hidden_dim=hidden_dim,
        probe_dim=32, dropout=0.0,
        lr=1e-3, weight_decay=0, max_epochs=5, patience=5, batch_size=8,
    )
    assert len(history["val_loss"]) > 0

    # 6. Test probe trigger
    probe_trigger = ScalarProbeTrigger(model, features)
    probe_curve = compute_operating_curve(
        probe_trigger, utt_ids, default_wers, careful_wers, num_thresholds=11
    )
    assert len(probe_curve["net_wers"]) == 11

    # 7. Test layer weights
    weights = model.layer_pool.get_layer_weights()
    assert len(weights) == num_layers
    assert abs(sum(weights) - 1.0) < 1e-5

    print("All E2E tiny checks passed!")


if __name__ == "__main__":
    test_e2e_tiny()
