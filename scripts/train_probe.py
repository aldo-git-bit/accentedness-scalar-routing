"""Train the accentedness scalar probe and produce routing scores."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.triggers.scalar_probe import AccentednessProbe, ScalarProbeTrigger
from accentedness_routing.triggers.train_probe import compute_calibration, train_probe


def load_features_and_targets(utterances, cfg) -> tuple[dict, dict]:
    """Load cached features and WER targets."""
    features_dir = Path(cfg["features"]["cache_dir"])
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]

    features = {}
    targets = {}

    for utt in utterances:
        uid = utt.utterance_id
        feat_path = features_dir / f"{uid}.pt"
        if not feat_path.exists():
            continue
        asr_result = load_cached(cache_dir, default_model, uid)
        if asr_result is None:
            continue

        features[uid] = torch.load(feat_path, weights_only=True)
        targets[uid] = asr_result["wer"]

    return features, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")

    # Load splits
    splits = {}
    for name in ["train", "val", "test"]:
        with open(data_dir / f"{name}_utterances.pkl", "rb") as f:
            splits[name] = pickle.load(f)
        print(f"Loaded {len(splits[name])} {name} utterances")

    # Load features and targets for each split
    split_features = {}
    split_targets = {}
    for name, utts in splits.items():
        feats, tgts = load_features_and_targets(utts, cfg)
        split_features[name] = feats
        split_targets[name] = tgts
        print(f"  {name}: {len(feats)} with features+targets")

    # Stack into tensors
    train_ids = list(split_features["train"].keys())
    val_ids = list(split_features["val"].keys())

    train_X = torch.stack([split_features["train"][uid] for uid in train_ids])
    train_y = torch.tensor([split_targets["train"][uid] for uid in train_ids], dtype=torch.float32)
    val_X = torch.stack([split_features["val"][uid] for uid in val_ids])
    val_y = torch.tensor([split_targets["val"][uid] for uid in val_ids], dtype=torch.float32)

    print(f"\nTraining: {train_X.shape[0]} samples, Validation: {val_X.shape[0]} samples")

    # Train
    probe_cfg = cfg["probe"]
    print("\nTraining probe...")
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

    # Report layer weights
    layer_weights = model.layer_pool.get_layer_weights()
    print("\nLayer weights (top 5):")
    ranked = sorted(enumerate(layer_weights), key=lambda x: -x[1])
    for i, (layer_idx, w) in enumerate(ranked[:5]):
        print(f"  Layer {layer_idx}: {w:.4f}")

    # Compute calibration from training set
    calibration = compute_calibration(model, train_X)
    print(f"\nCalibration: low={calibration['low']:.4f}, high={calibration['high']:.4f}")

    # Save model + metadata
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(probe_cfg["model_path"])

    torch.save({
        "model_state_dict": model.state_dict(),
        "calibration": calibration,
        "layer_weights": layer_weights,
        "history": history,
        "config": probe_cfg,
    }, model_path)
    print(f"Saved model to {model_path}")

    # Save training history
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-02-scalar-vs-baselines"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(exp_dir / "layer_weights.json", "w") as f:
        json.dump({
            "weights": [float(w) for w in layer_weights],
            "ranked": [[idx, float(w)] for idx, w in ranked[:10]],
        }, f, indent=2)


if __name__ == "__main__":
    main()
