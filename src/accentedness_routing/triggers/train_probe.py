"""Training loop for the accentedness probe."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

from accentedness_routing.triggers.scalar_probe import AccentednessProbe


def train_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    num_layers: int = 25,
    hidden_dim: int = 1024,
    probe_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 32,
    huber_delta: float = 0.1,
) -> tuple[AccentednessProbe, dict]:
    """Train the probe with early stopping.

    Args:
        train_features: (N_train, num_layers, hidden_dim)
        train_targets: (N_train,) — per-utterance WER from default model
        val_features: (N_val, num_layers, hidden_dim)
        val_targets: (N_val,)

    Returns:
        (trained model, training history dict)
    """
    model = AccentednessProbe(num_layers, hidden_dim, probe_dim, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=huber_delta)

    train_ds = TensorDataset(train_features, train_targets.unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_pearson_r": []}

    for epoch in range(max_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_features).squeeze(1)
            val_loss = criterion(val_pred, val_targets).item()

            # Pearson r
            pred_np = val_pred.numpy()
            target_np = val_targets.numpy()
            if np.std(pred_np) > 1e-8 and np.std(target_np) > 1e-8:
                r, _ = pearsonr(pred_np, target_np)
            else:
                r = 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_pearson_r"].append(r)

        print(
            f"  Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_r={r:.3f}"
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_loss"] = best_val_loss
    history["best_epoch"] = len(history["val_loss"]) - epochs_without_improvement

    return model, history


def compute_calibration(
    model: AccentednessProbe,
    features: torch.Tensor,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> dict:
    """Compute percentile-based calibration from training set predictions."""
    model.eval()
    with torch.no_grad():
        preds = model(features).squeeze(1).numpy()

    return {
        "low": float(np.percentile(preds, percentile_low)),
        "high": float(np.percentile(preds, percentile_high)),
    }
