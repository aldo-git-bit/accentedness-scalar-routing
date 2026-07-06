"""Extension 4: Three-way ablation with stats-pooled features.

All use champion target from Ext 1:
  1. mean+std: AccentednessProbe(hidden_dim=2048) on stats features
  2. mean-only: AccentednessProbe(hidden_dim=1024) retrained with champion target
  3. std-only: AccentednessProbe(hidden_dim=1024) on std columns only

Produces: models/probe_mean_std.pt, models/probe_mean_only.pt, models/probe_std_only.pt
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import cap_wer, escalation_gain
from accentedness_routing.triggers.train_probe import compute_calibration, train_probe


def load_targets(utterances, cfg, target_type="gain"):
    """Load targets using champion target type."""
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    targets = {}
    for utt in utterances:
        uid = utt.utterance_id
        d = load_cached(cache_dir, default_model, uid)
        c = load_cached(cache_dir, careful_model, uid)
        if d is None or c is None:
            continue
        if target_type == "gain":
            targets[uid] = escalation_gain(d["wer"], c["wer"])
        else:
            targets[uid] = cap_wer(d["wer"])
    return targets


def train_and_save(name, train_X, train_y, val_X, val_y, cfg, hidden_dim, model_path):
    """Train a probe variant and save."""
    probe_cfg = cfg.get("probe_gain", cfg["probe"])

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Input dim: {train_X.shape[-1]}, Hidden: {hidden_dim}")
    print(f"  Train: {train_X.shape[0]}, Val: {val_X.shape[0]}")
    print(f"{'='*60}")

    model, history = train_probe(
        train_X, train_y, val_X, val_y,
        num_layers=cfg["features"]["num_layers"],
        hidden_dim=hidden_dim,
        probe_dim=probe_cfg["hidden_dim"],
        dropout=probe_cfg["dropout"],
        lr=probe_cfg["lr"],
        weight_decay=probe_cfg["weight_decay"],
        max_epochs=probe_cfg["max_epochs"],
        patience=probe_cfg["patience"],
        batch_size=probe_cfg["batch_size"],
        huber_delta=probe_cfg["huber_delta"],
    )

    calibration = compute_calibration(model, train_X)
    layer_weights = model.layer_pool.get_layer_weights()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "calibration": calibration,
        "layer_weights": layer_weights,
        "history": history,
        "config": {**probe_cfg, "hidden_dim_input": hidden_dim},
        "variant": name,
    }, model_path)
    print(f"  Saved to {model_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Extension 4: Stats pooling ablation")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    stats_dir = Path(cfg["features_stats"]["cache_dir"])
    mean_dir = Path(cfg["features"]["cache_dir"])

    # Load splits
    with open(data_dir / "train_utterances.pkl", "rb") as f:
        train_utts = pickle.load(f)
    with open(data_dir / "val_utterances.pkl", "rb") as f:
        val_utts = pickle.load(f)

    # Determine champion target
    champion_path = Path(cfg["probe_gain"]["model_path"])
    target_type = "gain" if champion_path.exists() else "capped_wer"
    print(f"Target type: {target_type}")

    train_targets = load_targets(train_utts, cfg, target_type)
    val_targets = load_targets(val_utts, cfg, target_type)

    # Load stats features
    def load_features(utterances, feat_dir, targets_dict):
        feats = {}
        for utt in utterances:
            uid = utt.utterance_id
            if uid not in targets_dict:
                continue
            fp = feat_dir / f"{uid}.pt"
            if fp.exists():
                feats[uid] = torch.load(fp, weights_only=True)
        return feats

    train_stats = load_features(train_utts, stats_dir, train_targets)
    val_stats = load_features(val_utts, stats_dir, val_targets)
    train_mean = load_features(train_utts, mean_dir, train_targets)
    val_mean = load_features(val_utts, mean_dir, val_targets)

    print(f"Stats features: train={len(train_stats)}, val={len(val_stats)}")
    print(f"Mean features: train={len(train_mean)}, val={len(val_mean)}")

    models_dir = Path("models")

    # --- Variant 1: mean+std (hidden_dim=2048) ---
    if train_stats:
        common_train = sorted(set(train_stats.keys()) & set(train_targets.keys()))
        common_val = sorted(set(val_stats.keys()) & set(val_targets.keys()))

        train_X = torch.stack([train_stats[uid] for uid in common_train])
        train_y = torch.tensor([train_targets[uid] for uid in common_train], dtype=torch.float32)
        val_X = torch.stack([val_stats[uid] for uid in common_val])
        val_y = torch.tensor([val_targets[uid] for uid in common_val], dtype=torch.float32)

        train_and_save("mean_std", train_X, train_y, val_X, val_y, cfg,
                       hidden_dim=2048, model_path=models_dir / "probe_mean_std.pt")

        # --- Variant 3: std-only (hidden_dim=1024, take columns 1024:2048) ---
        train_X_std = train_X[:, :, 1024:]
        val_X_std = val_X[:, :, 1024:]
        train_and_save("std_only", train_X_std, train_y, val_X_std, val_y, cfg,
                       hidden_dim=1024, model_path=models_dir / "probe_std_only.pt")
    else:
        print("No stats features found. Run `make features-stats` first.")

    # --- Variant 2: mean-only retrained with champion target (hidden_dim=1024) ---
    if train_mean:
        common_train = sorted(set(train_mean.keys()) & set(train_targets.keys()))
        common_val = sorted(set(val_mean.keys()) & set(val_targets.keys()))

        train_X = torch.stack([train_mean[uid] for uid in common_train])
        train_y = torch.tensor([train_targets[uid] for uid in common_train], dtype=torch.float32)
        val_X = torch.stack([val_mean[uid] for uid in common_val])
        val_y = torch.tensor([val_targets[uid] for uid in common_val], dtype=torch.float32)

        train_and_save("mean_only", train_X, train_y, val_X, val_y, cfg,
                       hidden_dim=1024, model_path=models_dir / "probe_mean_only.pt")

    print("\nExtension 4 training complete.")


if __name__ == "__main__":
    main()
