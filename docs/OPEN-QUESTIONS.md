# Open Questions

## Resolved in Round 2

1. ~~Does the scalar generalize to accents not in the training set?~~ Not tested — the probe failed to beat confidence even on in-distribution accents.
2. ~~How sensitive is the probe to the choice of WavLM layer weighting initialization?~~ Edge layers (0, 24) dominate for gain prediction; mid layers (11-13) for accent classification. Layer weights are stable.
3. ~~Would a multi-task objective (WER regression + accent classification) improve routing?~~ Yes marginally — lambda=0.1 is the best learned trigger (area 0.011) but not significantly better than confidence (0.015).

## Resolved in Round 3

4. ~~What is the optimal escalation budget for production deployment?~~ Depends on model gap. At narrow gaps (small->large-v3), confidence-based routing at 10-20% escalation rate captures most benefit. At wider gaps, higher escalation rates are justified.
5. ~~Does widening the model gap create a regime where learned triggers outperform confidence?~~ Partially — the champion probe crosses confidence at tiny->large-v3 (area 0.014 vs 0.009) but the advantage is not statistically significant on 179 test utterances.
6. ~~Does temporal variability from WavLM help routing?~~ Improves classification AUC (+11 points) but not operating-curve area. Captures general difficulty, not escalation-specific difficulty.

## Open

7. Does the champion probe's advantage at wide gaps become significant with a larger test set (5-10x)?
8. Why does high classification AUC (0.78) not translate to proportional area-vs-random improvement? Is this a calibration issue or a geometric property of operating curves?
9. Would an accent-adapted careful model (e.g., Indian English finetune) expand headroom on the highest-gain accent slice?
10. Can segment-level or attention-weighted WavLM features capture escalation-specific difficulty that mean+std pooling misses?
