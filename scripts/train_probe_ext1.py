"""Extension 1: Train gain-target and capped-WER probes.

Two probes with identical AccentednessProbe architecture, different targets:
  1. gain_target: target = escalation_gain(wer_default, wer_careful) per utterance
  2. capped_wer: target = cap_wer(wer_default) (controlled comparison)

Produces: models/probe_gain.pt, models/probe_capped_wer.pt
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import cap_wer, escalation_gain
from accentedness_routing.triggers.train_probe import compute_calibration, train_probe


def load_split_data(utterances, cfg):
    """Load features and both models' WER for a split."""
    features_dir = Path(cfg["features"]["cache_dir"])
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    features = {}
    default_wers = {}
    careful_wers = {}

    for utt in utterances:
        uid = utt.utterance_id
        feat_path = features_dir / f"{uid}.pt"
        if not feat_path.exists():
            continue
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d is None or c is None:
            continue

        features[uid] = torch.load(feat_path, weights_only=True)
        default_wers[uid] = d["wer"]
        careful_wers[uid] = c["wer"]

    return features, default_wers, careful_wers


def train_and_save(
    name: str,
    train_features: dict,
    train_targets: dict,
    val_features: dict,
    val_targets: dict,
    cfg: dict,
    probe_cfg: dict,
    model_path: Path,
):
    """Train a probe and save checkpoint."""
    # Align IDs
    train_ids = sorted(train_features.keys() & train_targets.keys())
    val_ids = sorted(val_features.keys() & val_targets.keys())

    train_X = torch.stack([train_features[uid] for uid in train_ids])
    train_y = torch.tensor([train_targets[uid] for uid in train_ids], dtype=torch.float32)
    val_X = torch.stack([val_features[uid] for uid in val_ids])
    val_y = torch.tensor([val_targets[uid] for uid in val_ids], dtype=torch.float32)

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"  Target range: [{train_y.min():.3f}, {train_y.max():.3f}]")
    print(f"{'='*60}")

    model, history = train_probe(
        train_X, train_y, val_X, val_y,
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
        seed=cfg.get("seed", 42),
    )

    # Calibration
    calibration = compute_calibration(model, train_X)
    layer_weights = model.layer_pool.get_layer_weights()

    print(f"\n  Best val loss: {history['best_val_loss']:.4f}")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Calibration: [{calibration['low']:.4f}, {calibration['high']:.4f}]")

    # Save
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "calibration": calibration,
        "layer_weights": layer_weights,
        "history": history,
        "config": probe_cfg,
        "target_type": name,
    }, model_path)
    print(f"  Saved to {model_path}")

    return model, history, calibration, layer_weights


def main():
    parser = argparse.ArgumentParser(description="Extension 1: Train gain + capped-WER probes")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")

    # Load splits
    splits = {}
    for name in ["train", "val"]:
        with open(data_dir / f"{name}_utterances.pkl", "rb") as f:
            splits[name] = pickle.load(f)
        print(f"Loaded {len(splits[name])} {name} utterances")

    # Load features and WERs
    split_data = {}
    for name, utts in splits.items():
        feats, d_wers, c_wers = load_split_data(utts, cfg)
        split_data[name] = (feats, d_wers, c_wers)
        print(f"  {name}: {len(feats)} with features + both WERs")

    train_feats, train_d_wers, train_c_wers = split_data["train"]
    val_feats, val_d_wers, val_c_wers = split_data["val"]

    # --- Probe 1: gain_target ---
    gain_targets_train = {
        uid: escalation_gain(train_d_wers[uid], train_c_wers[uid])
        for uid in train_feats
    }
    gain_targets_val = {
        uid: escalation_gain(val_d_wers[uid], val_c_wers[uid])
        for uid in val_feats
    }

    train_and_save(
        "gain_target",
        train_feats, gain_targets_train,
        val_feats, gain_targets_val,
        cfg, cfg["probe_gain"],
        Path(cfg["probe_gain"]["model_path"]),
    )

    # --- Probe 2: capped_wer ---
    capped_targets_train = {uid: cap_wer(train_d_wers[uid]) for uid in train_feats}
    capped_targets_val = {uid: cap_wer(val_d_wers[uid]) for uid in val_feats}

    train_and_save(
        "capped_wer",
        train_feats, capped_targets_train,
        val_feats, capped_targets_val,
        cfg, cfg["probe_capped_wer"],
        Path(cfg["probe_capped_wer"]["model_path"]),
    )

    print("\nExtension 1 training complete.")


if __name__ == "__main__":
    main()
