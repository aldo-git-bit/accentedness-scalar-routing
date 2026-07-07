"""EXP-09: Headroom grid — full comparison per model pairing.

For each of the 6 pairings in {tiny, base, small} x {turbo, large-v3}:
  1. Load both models' ASR caches
  2. Build triggers: oracle, random, confidence, no_speech_prob, argmax_accent,
     champion_retrained (multitask lambda=0.1 retrained per pairing)
  3. Run full eval_common pipeline
  4. paired_bootstrap each trigger vs that pairing's confidence
  5. headroom_summary per pairing
  6. Save per-pairing metrics.json

Produces: experiments/EXP-09-headroom-grid/{pairing_slug}/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

from accentedness_routing.asr.cache import load_cached
from accentedness_routing.eval.eval_common import (
    bootstrap,
    cap_wer,
    decision_scorecard,
    escalation_gain,
    headroom_summary,
    operating_curve,
    paired_bootstrap,
    summarize,
)
from accentedness_routing.triggers.commonaccent import ArgmaxAccentTrigger
from accentedness_routing.triggers.confidence import ConfidenceTrigger
from accentedness_routing.triggers.hallucination import NoSpeechProbTrigger
from accentedness_routing.triggers.multitask_probe import MultiTaskProbe
from accentedness_routing.triggers.oracle import OracleTrigger
from accentedness_routing.triggers.random_trigger import RandomTrigger


def _model_slug(model_path: str) -> str:
    """Short slug for a model path."""
    return model_path.split("/")[-1]


def _pairing_slug(default_model: str, careful_model: str) -> str:
    return f"{_model_slug(default_model)}__{_model_slug(careful_model)}"


def load_asr_data(utterances, cache_dir: str, model_path: str):
    """Load ASR results for all utterances from cache.

    Returns dicts keyed by utterance_id: wer, logprobs, no_speech_probs,
    accent_map, speaker_map, hypotheses.
    """
    wers = {}
    logprobs = {}
    no_speech_probs = {}
    accent_map = {}
    speaker_map = {}
    hypotheses = {}

    for utt in utterances:
        uid = utt.utterance_id
        result = load_cached(cache_dir, model_path, uid)
        if result is None:
            continue
        wers[uid] = result["wer"]
        logprobs[uid] = result.get("avg_logprob", 0.0)
        no_speech_probs[uid] = result.get("no_speech_prob", 0.0)
        accent_map[uid] = utt.accent
        speaker_map[uid] = utt.speaker
        hypotheses[uid] = result.get("text", "")

    return wers, logprobs, no_speech_probs, accent_map, speaker_map, hypotheses


def retrain_champion(
    train_features: dict[str, torch.Tensor],
    train_gains: dict[str, float],
    train_accents: dict[str, str],
    val_features: dict[str, torch.Tensor],
    val_gains: dict[str, float],
    val_accents: dict[str, str],
    cfg: dict,
    save_path: Path | None = None,
) -> tuple[MultiTaskProbe, dict]:
    """Retrain multitask lambda=0.1 probe on a specific pairing's gain target.

    Returns (model, calibration).
    """
    all_accents = sorted(set(list(train_accents.values()) + list(val_accents.values())))
    accent_to_idx = {accent: i for i, accent in enumerate(all_accents)}

    # Build tensors
    train_ids = sorted(set(train_features.keys()) & set(train_gains.keys()))
    val_ids = sorted(set(val_features.keys()) & set(val_gains.keys()))

    if len(train_ids) < 10 or len(val_ids) < 5:
        return None, None

    train_X = torch.stack([train_features[uid] for uid in train_ids])
    train_reg_y = torch.tensor([train_gains[uid] for uid in train_ids], dtype=torch.float32)
    train_cls_y = torch.tensor([accent_to_idx.get(train_accents[uid], 0)
                                for uid in train_ids], dtype=torch.long)

    val_X = torch.stack([val_features[uid] for uid in val_ids])
    val_reg_y = torch.tensor([val_gains[uid] for uid in val_ids], dtype=torch.float32)
    val_cls_y = torch.tensor([accent_to_idx.get(val_accents[uid], 0)
                              for uid in val_ids], dtype=torch.long)

    probe_cfg = cfg.get("probe_gain", cfg["probe"])
    lam = 0.1

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

        # Validate
        model.eval()
        with torch.no_grad():
            val_reg_pred, val_cls_pred = model(val_X)
            val_loss_reg = reg_criterion(val_reg_pred, val_reg_y.unsqueeze(1)).item()
            val_loss_cls = cls_criterion(val_cls_pred, val_cls_y).item()
            val_loss = (1 - lam) * val_loss_reg + lam * val_loss_cls

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= probe_cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibration on regression head
    model.eval()
    with torch.no_grad():
        train_preds = model.predict_score(train_X).squeeze(1).numpy()
    calibration = {
        "low": float(np.percentile(train_preds, 2.0)),
        "high": float(np.percentile(train_preds, 98.0)),
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "calibration": calibration,
            "accent_to_idx": accent_to_idx,
            "lambda": lam,
        }, save_path)

    return model, calibration


def main():
    parser = argparse.ArgumentParser(description="EXP-09: Headroom grid evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path("data")
    cache_dir = cfg["asr"]["cache_dir"]
    features_dir = Path(cfg["features"]["cache_dir"])
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = Path(cfg["output"]["experiments_dir"]) / "EXP-09-headroom-grid"
    exp_dir.mkdir(parents=True, exist_ok=True)

    grid = cfg.get("asr_grid", {})
    default_models = grid.get("default_models", [cfg["asr"]["default_model"]])
    careful_models = grid.get("careful_models", [cfg["asr"]["careful_model"]])

    # Load splits
    splits = {}
    for name in ["train", "val", "test"]:
        pkl_path = data_dir / f"{name}_utterances.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                splits[name] = pickle.load(f)
            print(f"Loaded {len(splits[name])} {name} utterances")

    test_utts = splits.get("test", [])
    train_utts = splits.get("train", [])
    val_utts = splits.get("val", [])

    # Load WavLM features for champion retraining
    def load_features(utterances):
        feats = {}
        for utt in utterances:
            fp = features_dir / f"{utt.utterance_id}.pt"
            if fp.exists():
                feats[utt.utterance_id] = torch.load(fp, weights_only=True)
        return feats

    print("Loading WavLM features...")
    train_features = load_features(train_utts)
    val_features = load_features(val_utts)
    test_features = load_features(test_utts)
    print(f"  train={len(train_features)}, val={len(val_features)}, test={len(test_features)}")

    # Iterate over all pairings
    pairings = [(d, c) for d in default_models for c in careful_models]
    print(f"\nGrid: {len(pairings)} pairings")

    for default_model, careful_model in pairings:
        slug = _pairing_slug(default_model, careful_model)
        print(f"\n{'='*60}")
        print(f"Pairing: {slug}")
        print(f"{'='*60}")

        pairing_dir = exp_dir / slug
        pairing_dir.mkdir(parents=True, exist_ok=True)

        # Load ASR data for both models
        (wer_default, logprobs_default, nsp_default,
         accent_map, speaker_map, hyp_default) = load_asr_data(
            test_utts, cache_dir, default_model)
        (wer_careful, _, _, _, _, _) = load_asr_data(
            test_utts, cache_dir, careful_model)

        # Intersect utterance IDs
        utt_ids = sorted(set(wer_default.keys()) & set(wer_careful.keys()))
        if len(utt_ids) == 0:
            print(f"  SKIP: no common utterances")
            continue

        wd = {uid: wer_default[uid] for uid in utt_ids}
        wc = {uid: wer_careful[uid] for uid in utt_ids}
        gains = {uid: escalation_gain(wd[uid], wc[uid]) for uid in utt_ids}

        # Per-accent mean capped WER for argmax trigger
        accent_wer_sums = defaultdict(float)
        accent_counts = defaultdict(int)
        for uid in utt_ids:
            accent_wer_sums[accent_map[uid]] += cap_wer(wd[uid])
            accent_counts[accent_map[uid]] += 1
        accent_mean_wers = {a: accent_wer_sums[a] / accent_counts[a]
                            for a in accent_wer_sums}

        # Build triggers
        oracle = OracleTrigger(wd, wc)
        random_trigger = RandomTrigger(seed=cfg.get("seed", 42))
        confidence = ConfidenceTrigger({uid: logprobs_default[uid] for uid in utt_ids})
        no_speech = NoSpeechProbTrigger({uid: nsp_default[uid] for uid in utt_ids})
        argmax = ArgmaxAccentTrigger(
            {uid: accent_map[uid] for uid in utt_ids}, accent_mean_wers)

        all_triggers = {
            "oracle": {uid: oracle.score(uid) for uid in utt_ids},
            "random": {uid: random_trigger.score(uid) for uid in utt_ids},
            "confidence": {uid: confidence.score(uid) for uid in utt_ids},
            "no_speech_prob": {uid: no_speech.score(uid) for uid in utt_ids},
            "argmax_accent": {uid: argmax.score(uid) for uid in utt_ids},
        }

        # Champion retrained: multitask lambda=0.1 on this pairing's gain target
        # Build train/val gains for this pairing
        train_asr_d = {}
        train_asr_c = {}
        for utt in train_utts:
            uid = utt.utterance_id
            d = load_cached(cache_dir, default_model, uid)
            c = load_cached(cache_dir, careful_model, uid)
            if d is not None and c is not None:
                train_asr_d[uid] = d["wer"]
                train_asr_c[uid] = c["wer"]

        val_asr_d = {}
        val_asr_c = {}
        for utt in val_utts:
            uid = utt.utterance_id
            d = load_cached(cache_dir, default_model, uid)
            c = load_cached(cache_dir, careful_model, uid)
            if d is not None and c is not None:
                val_asr_d[uid] = d["wer"]
                val_asr_c[uid] = c["wer"]

        train_gains = {uid: escalation_gain(train_asr_d[uid], train_asr_c[uid])
                       for uid in train_asr_d if uid in train_asr_c}
        val_gains_dict = {uid: escalation_gain(val_asr_d[uid], val_asr_c[uid])
                         for uid in val_asr_d if uid in val_asr_c}

        train_accents = {utt.utterance_id: utt.accent for utt in train_utts}
        val_accents = {utt.utterance_id: utt.accent for utt in val_utts}

        champion_path = models_dir / f"champion_{slug}.pt"
        print(f"  Retraining champion for {slug}...")
        model, calibration = retrain_champion(
            train_features, train_gains, train_accents,
            val_features, val_gains_dict, val_accents,
            cfg, save_path=champion_path,
        )

        if model is not None:
            model.eval()
            low, high = calibration["low"], calibration["high"]
            rng = high - low if high - low > 1e-8 else 1.0
            champion_scores = {}
            with torch.no_grad():
                for uid in utt_ids:
                    if uid in test_features:
                        raw = model.predict_score(test_features[uid].unsqueeze(0)).item()
                        normed = (raw - low) / rng
                        champion_scores[uid] = max(0.0, min(1.0, normed))
                    else:
                        champion_scores[uid] = 0.5  # fallback

            all_triggers["champion_retrained"] = champion_scores
            print(f"  Champion retrained: {len(champion_scores)} scores")
        else:
            print(f"  Champion retraining failed (insufficient data)")

        # Full eval_common pipeline per trigger
        random_scores = all_triggers["random"]
        confidence_scores = all_triggers["confidence"]
        results: dict = {"triggers": {}, "pairing": slug}

        for trig_name, trig_scores in all_triggers.items():
            print(f"  Scoring: {trig_name}")
            curve = operating_curve(trig_scores, wd, wc)
            random_curve = operating_curve(random_scores, wd, wc)
            summary = summarize(curve, random_curve)

            scorecard_0 = decision_scorecard(trig_scores, gains, tau=0.0)
            scorecard_05 = decision_scorecard(trig_scores, gains, tau=0.05)

            boot = bootstrap(
                trig_scores, wd, wc,
                n=args.bootstrap_n, seed=cfg.get("seed", 42),
                random_scores=random_scores,
            )

            entry = {
                "summary": summary,
                "scorecard_tau_0.00": scorecard_0,
                "scorecard_tau_0.05": scorecard_05,
                "bootstrap_ci": boot["ci"],
                "curve_bands": boot["curve_bands"],
            }

            # Paired bootstrap vs confidence
            if trig_name != "confidence" and trig_name != "random":
                paired = paired_bootstrap(
                    trig_scores, confidence_scores,
                    wd, wc,
                    n=args.bootstrap_n, seed=cfg.get("seed", 42),
                    random_scores=random_scores,
                )
                entry["paired_vs_confidence"] = paired

            results["triggers"][trig_name] = entry

            for k, v in summary.items():
                if v is not None:
                    print(f"    {k}: {v:.4f}")

        # Headroom summary
        results["headroom"] = headroom_summary(
            wd, wc, all_triggers["oracle"], random_scores)
        print(f"  Headroom: oracle_avr={results['headroom']['oracle_area_vs_random']:.4f}, "
              f"wer_gap={results['headroom']['wer_gap']:.4f}")

        # Save
        with open(pairing_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved: {pairing_dir / 'metrics.json'}")

    print(f"\nAll pairings complete. Results in {exp_dir}")


if __name__ == "__main__":
    main()
