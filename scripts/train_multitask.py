"""Extension 5: Train multi-task probe with lambda sweep.

Combined loss: (1-lambda) * HuberLoss(regression) + lambda * CrossEntropyLoss(accent).
Sweep lambda in {0.0, 0.1, 0.3, 1.0}.

Produces: models/probe_multitask_lam{lambda}.pt
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

import yaml

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import cap_wer, escalation_gain
from accentedness_routing.triggers.multitask_probe import MultiTaskProbe


def main():
    parser = argparse.ArgumentParser(description="Extension 5: Multi-task lambda sweep")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(data_dir / "train_utterances.pkl", "rb") as f:
        train_utts = pickle.load(f)
    with open(data_dir / "val_utterances.pkl", "rb") as f:
        val_utts = pickle.load(f)

    features_dir = Path(cfg["features"]["cache_dir"])
    cache_dir = cfg["asr"]["cache_dir"]
    default_model = cfg["asr"]["default_model"]
    careful_model = cfg["asr"]["careful_model"]

    # Build accent label mapping
    all_accents = sorted(set(utt.accent for utt in train_utts + val_utts))
    accent_to_idx = {accent: i for i, accent in enumerate(all_accents)}
    print(f"Accents: {accent_to_idx}")

    # Determine champion target
    champion_path = Path(cfg["probe_gain"]["model_path"])
    target_type = "gain" if champion_path.exists() else "capped_wer"
    print(f"Target type: {target_type}")

    def load_data(utterances):
        feats, reg_targets, cls_targets = [], [], []
        for utt in utterances:
            uid = utt.utterance_id
            fp = features_dir / f"{uid}.pt"
            if not fp.exists():
                continue
            d = load_cached(cache_dir, default_model, uid)
            c = load_cached(cache_dir, careful_model, uid)
            if d is None or c is None:
                continue

            feats.append(torch.load(fp, weights_only=True))
            if target_type == "gain":
                reg_targets.append(escalation_gain(d["wer"], c["wer"]))
            else:
                reg_targets.append(cap_wer(d["wer"]))
            cls_targets.append(accent_to_idx[utt.accent])

        return (torch.stack(feats),
                torch.tensor(reg_targets, dtype=torch.float32),
                torch.tensor(cls_targets, dtype=torch.long))

    train_X, train_reg_y, train_cls_y = load_data(train_utts)
    val_X, val_reg_y, val_cls_y = load_data(val_utts)
    print(f"Train: {train_X.shape[0]}, Val: {val_X.shape[0]}")

    probe_cfg = cfg.get("probe_gain", cfg["probe"])
    lambdas = [0.0, 0.1, 0.3, 1.0]
    all_results = {"lambdas": {}, "accent_to_idx": accent_to_idx}

    for lam in lambdas:
        print(f"\n{'='*60}")
        print(f"Lambda = {lam}")
        print(f"{'='*60}")

        model = MultiTaskProbe(
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
        reg_criterion = nn.HuberLoss(delta=probe_cfg["huber_delta"])
        cls_criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(train_X, train_reg_y.unsqueeze(1), train_cls_y)
        train_loader = DataLoader(train_ds, batch_size=probe_cfg["batch_size"], shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        history: dict = {"train_loss": [], "val_loss": [], "val_pearson_r": [],
                         "val_accuracy": []}

        for epoch in range(probe_cfg["max_epochs"]):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, reg_batch, cls_batch in train_loader:
                optimizer.zero_grad()
                reg_pred, cls_pred = model(x_batch)
                loss_reg = reg_criterion(reg_pred, reg_batch)
                loss_cls = cls_criterion(cls_pred, cls_batch)
                loss = (1 - lam) * loss_reg + lam * loss_cls
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            # Validate
            model.eval()
            with torch.no_grad():
                val_reg_pred, val_cls_pred = model(val_X)
                val_loss_reg = reg_criterion(val_reg_pred, val_reg_y.unsqueeze(1)).item()
                val_loss_cls = cls_criterion(val_cls_pred, val_cls_y).item()
                val_loss = (1 - lam) * val_loss_reg + lam * val_loss_cls

                # Pearson r on regression
                pred_np = val_reg_pred.squeeze(1).numpy()
                target_np = val_reg_y.numpy()
                if np.std(pred_np) > 1e-8 and np.std(target_np) > 1e-8:
                    r, _ = pearsonr(pred_np, target_np)
                else:
                    r = 0.0

                # Accuracy
                val_acc = float((val_cls_pred.argmax(dim=1) == val_cls_y).float().mean())

            history["train_loss"].append(float(avg_train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_pearson_r"].append(float(r))
            history["val_accuracy"].append(float(val_acc))

            print(f"  Epoch {epoch+1:3d}: loss={avg_train_loss:.4f}  "
                  f"val={val_loss:.4f}  r={r:.3f}  acc={val_acc:.3f}")

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

        # Calibration on regression head
        model.eval()
        with torch.no_grad():
            train_preds = model.predict_score(train_X).squeeze(1).numpy()
        calibration = {
            "low": float(np.percentile(train_preds, 2.0)),
            "high": float(np.percentile(train_preds, 98.0)),
        }

        # Save
        lam_str = f"{lam:.1f}".replace(".", "")
        model_path = models_dir / f"probe_multitask_lam{lam_str}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "calibration": calibration,
            "layer_weights": layer_weights,
            "history": history,
            "config": {**probe_cfg, "lambda": lam},
            "accent_to_idx": accent_to_idx,
            "lambda": lam,
        }, model_path)
        print(f"  Saved to {model_path}")

        all_results["lambdas"][str(lam)] = {
            "best_val_loss": float(best_val_loss),
            "final_pearson_r": history["val_pearson_r"][-1],
            "final_accuracy": history["val_accuracy"][-1],
            "layer_weights": layer_weights,
        }

    # Save summary
    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-08-extension5-multitask"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "training_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTraining summary saved to {exp_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
