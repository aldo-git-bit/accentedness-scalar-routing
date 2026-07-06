"""Extension 2: Train champion probe on {25%, 50%, 75%, 100%} of train fold.

Subsamples BY SPEAKER (not utterance) to preserve speaker-disjointness.
Records val AUC and Pearson r at each fraction.

Produces: experiments/EXP-05-extension2-diagnostics/learning_curve.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import cap_wer, decision_scorecard, escalation_gain
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger
from accentedness_routing.triggers.train_probe import compute_calibration, train_probe


def main():
    parser = argparse.ArgumentParser(description="Extension 2: Learning curve experiment")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-05-extension2-diagnostics"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(data_dir / "train_utterances.pkl", "rb") as f:
        train_utts = pickle.load(f)
    with open(data_dir / "val_utterances.pkl", "rb") as f:
        val_utts = pickle.load(f)

    # Load splits manifest for speaker info
    with open(data_dir / "splits_manifest.json") as f:
        manifest = json.load(f)

    print(f"Train: {len(train_utts)}, Val: {len(val_utts)}")

    # Load features and WERs
    features_dir = Path(cfg["features"]["cache_dir"])
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    # Determine champion target type from Ext 1 results
    # Default to gain_target; override if probe_gain model exists
    champion_model_path = Path(cfg["probe_gain"]["model_path"])
    if champion_model_path.exists():
        target_type = "gain"
        print("Using gain target (champion from Ext 1)")
    else:
        target_type = "capped_wer"
        print("Using capped_wer target (probe_gain not found)")

    def load_data(utterances):
        feats, targets, accents, speakers = {}, {}, {}, {}
        for utt in utterances:
            uid = utt.utterance_id
            fp = features_dir / f"{uid}.pt"
            if not fp.exists():
                continue
            d = load_cached(cache_dir, default_model, uid)
            c = load_cached(cache_dir, careful_model, uid)
            if d is None or c is None:
                continue

            feats[uid] = torch.load(fp, weights_only=True)
            if target_type == "gain":
                targets[uid] = escalation_gain(d["wer"], c["wer"])
            else:
                targets[uid] = cap_wer(d["wer"])
            accents[uid] = utt.accent
            speakers[uid] = utt.speaker

        return feats, targets, accents, speakers

    train_feats, train_targets, train_accents, train_speakers = load_data(train_utts)
    val_feats, val_targets, val_accents, val_speakers = load_data(val_utts)

    print(f"  Train with data: {len(train_feats)}, Val: {len(val_feats)}")

    # Prepare val tensors (fixed across all fractions)
    val_ids = sorted(val_feats.keys())
    val_X = torch.stack([val_feats[uid] for uid in val_ids])
    val_y = torch.tensor([val_targets[uid] for uid in val_ids], dtype=torch.float32)

    # Compute val gains for scorecard
    val_default_wers, val_careful_wers = {}, {}
    for utt in val_utts:
        uid = utt.utterance_id
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d is not None and c is not None and uid in val_feats:
            val_default_wers[uid] = d["wer"]
            val_careful_wers[uid] = c["wer"]
    val_gains = {uid: escalation_gain(val_default_wers[uid], val_careful_wers[uid])
                 for uid in val_ids if uid in val_default_wers}

    # Group train utterances by speaker, then by accent
    accent_speakers: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for uid in train_feats:
        accent_speakers[train_accents[uid]][train_speakers[uid]].append(uid)

    rng = np.random.RandomState(cfg.get("seed", 42))
    fractions = [0.25, 0.50, 0.75, 1.00]
    results = {"fractions": [], "target_type": target_type}

    probe_cfg = cfg.get("probe_gain", cfg["probe"])

    for frac in fractions:
        print(f"\n{'='*60}")
        print(f"Fraction: {frac:.0%}")
        print(f"{'='*60}")

        # Subsample by speaker per accent
        selected_uids = []
        for accent, speakers_dict in sorted(accent_speakers.items()):
            speaker_list = sorted(speakers_dict.keys())
            n_select = max(1, int(np.floor(frac * len(speaker_list))))
            chosen = rng.choice(speaker_list, size=n_select, replace=False).tolist()
            for spk in chosen:
                selected_uids.extend(speakers_dict[spk])

        selected_uids = sorted(set(selected_uids) & set(train_feats.keys()))

        train_X = torch.stack([train_feats[uid] for uid in selected_uids])
        train_y_tensor = torch.tensor([train_targets[uid] for uid in selected_uids],
                                      dtype=torch.float32)

        print(f"  Selected {len(selected_uids)} utterances")

        model, history = train_probe(
            train_X, train_y_tensor, val_X, val_y,
            num_layers=cfg["features"]["num_layers"],
            hidden_dim=cfg["features"]["hidden_dim"],
            probe_dim=probe_cfg["hidden_dim"],
            dropout=probe_cfg["dropout"],
            lr=probe_cfg["lr"],
            weight_decay=probe_cfg["weight_decay"],
            max_epochs=probe_cfg["max_epochs"],
            patience=probe_cfg["patience"],
            batch_size=probe_cfg["batch_size"],
            huber_delta=probe_cfg["huber_delta"],
        )

        # Compute val scorecard
        calibration = compute_calibration(model, train_X)
        trigger = ScalarProbeTrigger(model, val_feats, calibration)
        val_scores = {uid: trigger.score(uid) for uid in val_ids if uid in val_feats}

        scorecard = decision_scorecard(val_scores, val_gains, tau=0.0)

        frac_result = {
            "fraction": frac,
            "n_utterances": len(selected_uids),
            "best_val_loss": history["best_val_loss"],
            "best_epoch": history["best_epoch"],
            "final_val_pearson_r": history["val_pearson_r"][-1] if history["val_pearson_r"] else None,
            "val_auc": scorecard.get("auc"),
            "val_ap": scorecard.get("ap"),
            "val_pearson_r": scorecard.get("pearson_r"),
            "val_spearman_r": scorecard.get("spearman_r"),
            "layer_weights": model.layer_pool.get_layer_weights(),
        }
        results["fractions"].append(frac_result)

        print(f"  Val AUC: {frac_result['val_auc']}")
        print(f"  Val Pearson r: {frac_result['val_pearson_r']}")

    # Save
    with open(exp_dir / "learning_curve.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nLearning curve saved to {exp_dir / 'learning_curve.json'}")


if __name__ == "__main__":
    main()
