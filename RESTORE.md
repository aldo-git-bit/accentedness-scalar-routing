# Restoring this project

This repo was retired on 2026-10-25. The code, writeups, experiment results,
probe checkpoints, and ASR transcript cache are all in git. Everything else
was either archived off-repo or can be re-downloaded. This file says which is
which and how to get back to a working checkout.

## What lives where

| Location | Contents | Size |
|---|---|---|
| **This repo (git)** | Code, configs, docs, writeups, `experiments/` (reports, figures, `metrics.json`), `models/*.pt` (14 probe/classifier checkpoints), `data/asr_cache/` (Whisper transcripts, all models), `data/acoustic_features_cache/`, `data/splits_manifest.json`, `data/utterance_index.json` | ~47 MB |
| **Backup archive** `accentedness-scalar-routing-archive-2026-10-25.zip` | `data/{train,val,test}_utterances.pkl` (selected EdAcc audio), `data/features_cache/` (WavLM mean-pool), `data/features_cache_stats/` (WavLM mean+std), `.claude/settings.local.json` | 685 MB zipped |
| **Re-download** (public, automatic) | Python env (`.venv/`), EdAcc dataset, Whisper/WavLM model weights | ~13 GB |

Answering questions about the results needs only the repo: read
`README.md`, `writeup/writeup-v4-detailed.md`, and `experiments/COMPARISON.md`.

## Requirements

- Apple Silicon Mac (mlx-whisper)
- Python 3.10–3.12 (`pyproject.toml` pins `<3.13`; uv fetches one if needed)
- [uv](https://docs.astral.sh/uv/), ffmpeg, ~24 GB RAM
- ~15 GB free disk for downloads

## Step 1 — code and environment

```bash
git clone https://github.com/aldo-git-bit/accentedness-scalar-routing.git
cd accentedness-scalar-routing
make setup        # uv sync, from uv.lock
make smoke        # downloads WavLM + a Whisper model, checks they load
```

## Step 2 — restore the archive (recommended)

Unzip at the repo root; paths inside the zip are repo-relative.

```bash
unzip /path/to/accentedness-scalar-routing-archive-2026-10-25.zip -d .
```

With the archive restored, every step in the README's pipeline tables hits
cache and returns instantly — `make reproduce` and `make reproduce-v3`
re-run evaluation without decoding audio or extracting features.

## Step 2 (alternative) — rebuild without the archive

If the archive is lost, regenerate it from EdAcc. This is slow (hours on a
laptop: WavLM-large runs on CPU).

```bash
make data            # downloads EdAcc (~6.5 GB) and rebuilds the .pkl splits
make features        # WavLM mean-pool features -> data/features_cache/
make features-stats  # WavLM mean+std features  -> data/features_cache_stats/
```

`make data` re-selects utterances with `seed: 42` from `configs/default.yaml`.
Check the result against the committed `data/splits_manifest.json` and
`data/utterance_index.json`; if EdAcc has changed upstream, those two files
are the record of the original selection. ASR does not need re-running — the
transcripts are already in `data/asr_cache/`.

## Downloads, for reference

All fetched automatically into `~/.cache/huggingface/` on first use.

| What | Hugging Face ID | Size | Used by |
|---|---|---|---|
| EdAcc dataset | `edinburghcstr/edacc` | 6.5 GB | `make data` |
| WavLM-large | `microsoft/wavlm-large` | 2.3 GB | features, EXP-13 fine-tuning |
| Whisper tiny / base / small | `mlx-community/whisper-{tiny,base,small}-mlx` | 0.7 GB | ASR (cached) |
| Whisper large-v3 | `mlx-community/whisper-large-v3-mlx` | 2.9 GB | ASR (cached) |
| Whisper large-v3-turbo | `mlx-community/whisper-large-v3-turbo` | 1.5 GB | ASR (cached) |
| Accent-adapted Whisper | `Tejveer12/Indian-Accent-English-Whisper-Finetuned` | 3.0 GB | `make ext-adapted` only |

Whisper weights are only needed to decode new audio; existing transcripts are
cached in git. If a dataset becomes gated, copy `.env.example` to `.env` and
set `HF_TOKEN`.

## Where work stopped

The last active work was EXP-13 (WavLM-large DDP fine-tuning on a 2-GPU box),
on branch `exp13-wavlm-ddp`, now merged into `main`. See
`models/CHECKPOINT-NOTES.md` for which checkpoint is current and why.
