"""Extension 2: Train accent classifier for layer weight comparison.

Trains AccentClassifier with CrossEntropyLoss, same optimizer settings.
Reports accuracy, macro-F1, and saves learned layer weights.

Produces: models/accent_classifier.pt,
          experiments/EXP-05-extension2-diagnostics/accent_classifier.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.triggers.accent_probe import AccentClassifier


def main():
    parser = argparse.ArgumentParser(description="Extension 2: Train accent classifier")
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

    features_dir = Path(cfg["features"]["cache_dir"])

    # Build accent label mapping
    all_accents = sorted(set(utt.accent for utt in train_utts + val_utts))
    accent_to_idx = {accent: i for i, accent in enumerate(all_accents)}
    print(f"Accents: {accent_to_idx}")

    def load_data(utterances):
        feats, labels = [], []
        for utt in utterances:
            uid = utt.utterance_id
            fp = features_dir / f"{uid}.pt"
            if not fp.exists():
                continue
            feats.append(torch.load(fp, weights_only=True))
            labels.append(accent_to_idx[utt.accent])
        return torch.stack(feats), torch.tensor(labels, dtype=torch.long)

    train_X, train_y = load_data(train_utts)
    val_X, val_y = load_data(val_utts)
    print(f"Train: {train_X.shape[0]}, Val: {val_X.shape[0]}")

    # Train
    probe_cfg = cfg["probe"]
    model = AccentClassifier(
        num_accents=len(all_accents),
        num_layers=cfg["features"]["num_layers"],
        hidden_dim=cfg["features"]["hidden_dim"],
        probe_dim=probe_cfg["hidden_dim"],
        dropout=probe_cfg["dropout"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=probe_cfg["lr"],
        weight_decay=probe_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=probe_cfg["batch_size"], shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}

    for epoch in range(probe_cfg["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_y).item()
            val_preds = val_logits.argmax(dim=1).numpy()
            val_true = val_y.numpy()
            accuracy = float((val_preds == val_true).mean())
            macro_f1 = float(f1_score(val_true, val_preds, average="macro"))

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(accuracy)
        history["val_macro_f1"].append(macro_f1)

        print(f"  Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  acc={accuracy:.3f}  F1={macro_f1:.3f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= probe_cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    layer_weights = model.layer_pool.get_layer_weights()

    # Final val metrics
    model.eval()
    with torch.no_grad():
        val_logits = model(val_X)
        val_preds = val_logits.argmax(dim=1).numpy()
        val_true = val_y.numpy()

    final_accuracy = float((val_preds == val_true).mean())
    final_f1 = float(f1_score(val_true, val_preds, average="macro"))

    print(f"\nFinal: accuracy={final_accuracy:.3f}, macro-F1={final_f1:.3f}")
    print(f"Top layer weights: {sorted(enumerate(layer_weights), key=lambda x: -x[1])[:5]}")

    # Save model
    model_path = Path("models/accent_classifier.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "layer_weights": layer_weights,
        "accent_to_idx": accent_to_idx,
        "history": history,
    }, model_path)
    print(f"Saved model to {model_path}")

    # Save results
    results = {
        "accuracy": final_accuracy,
        "macro_f1": final_f1,
        "layer_weights": layer_weights,
        "accent_to_idx": accent_to_idx,
        "history": history,
        "best_val_loss": float(best_val_loss),
    }
    with open(exp_dir / "accent_classifier.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {exp_dir / 'accent_classifier.json'}")


if __name__ == "__main__":
    main()
