# Technical Decisions Log

| # | Decision | Choice | Rationale | Date |
|---|----------|--------|-----------|------|
| 1 | ASR pair | whisper-small → whisper-large-v3 | Max WER gap for routing headroom | 2025-07-05 |
| 2 | Scalar target | Base-model per-utterance WER | Direct difficulty signal; no careful-model transcripts needed for training | 2025-07-05 |
| 3 | Subset size | 6 accents × 150 utt ≈ 900 | Fits in 24 GB; full pipeline < 60 min | 2025-07-05 |
| 4 | WavLM loading | .from_pretrained().to("cpu") | Avoids device_map issues on Apple Silicon | 2025-07-05 |
| 5 | Split strategy | Speaker-disjoint 60/20/20 | Prevents speaker leakage | 2025-07-05 |
| 6 | Probe loss | HuberLoss(delta=0.1) | Robust to high-WER outliers | 2025-07-05 |
