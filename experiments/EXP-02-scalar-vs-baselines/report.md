# Experiment Report: Scalar Probe vs Baselines

## Operating Curve Summary

| Trigger | WER@10% | WER@20% | WER@30% | WER@50% | Area vs Random |
|---------|---------|---------|---------|---------|----------------|
| oracle | 0.7574 | 0.7394 | 0.7319 | 0.7349 | 0.6539 |
| random | 2.4539 | 2.4445 | 1.2142 | 1.1982 | 0.0000 |
| confidence | 2.9389 | 2.9235 | 2.9151 | 2.9102 | -1.3298 |
| argmax_accent | N/A | 1.2791 | 1.0781 | 0.7827 | 0.4543 |
| scalar_probe | 1.7222 | 0.9995 | 0.7965 | 0.7778 | 0.4523 |

## Per-Accent WER

| Accent | Default WER | Careful WER | Escalation Gain |
|--------|-------------|-------------|-----------------|
| Indian English | 8.9643 | 0.4030 | 8.5613 |
| Irish English | 0.3054 | 0.2900 | 0.0154 |
| Mainstream US English | 5.1398 | 3.1291 | 2.0107 |
| Nigerian English | 0.3970 | 0.3918 | 0.0052 |
| Scottish English | 0.3320 | 0.2597 | 0.0723 |
| Southern British English | 1.6709 | 0.3549 | 1.3160 |

## Key Result

Scalar probe does **not** dominate the argmax accent baseline.

## Speaker Leakage Check

- MI(score, speaker) = 0.3197
- MI(score, accent) = 0.2219
- Ratio = 1.44

## Figures

- [Operating Curves](figures/operating_curves.png)
- [Layer Weights](figures/layer_weights.png)
- [Score Distributions](figures/score_distributions.png)
- [Per-Accent WER](figures/per_accent_wer.png)
