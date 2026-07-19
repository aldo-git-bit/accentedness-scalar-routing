# Raw baseline is hostage to hallucination outliers

`curves.json` / `summaries.json` in this directory use **uncapped** WER
(`scripts/run_baselines.py` reads `d_result["wer"]` / `c_result["wer"]`
straight from `data/asr_cache/`, no ceiling). That makes the aggregate
numbers dominated by a small number of hallucinated transcriptions rather
than by routing quality.

## Reproduced 2026-07-19, exp13-wavlm-ddp branch (`make data asr features baselines`)

- 61/900 utterances in the whisper-small ASR cache have WER > 1.0
  (repetition-loop hallucinations), out of the full 900-utterance dataset.
- Max WER: **223.0** (one utterance). Mean uncapped WER: 6.031 over those
  900; on the 179-utterance test split specifically, mean default WER
  comes out to **2.9621** — i.e. the "average" utterance appears to be
  transcribed nearly 3x worse than the reference, which is impossible for
  a well-behaved WER and is purely an artifact of a handful of extreme
  outliers dragging the mean.
- Effect on trigger ranking: with uncapped WER, `confidence`'s
  area-vs-random is **-1.3298** (i.e. confidence looks *worse than
  random*). This is the exact failure mode described in
  `docs/DIAGNOSIS-pilot-to-v2.md` and the README's Finding #1.

## The fix and the corrected numbers

`scripts/rescore_pilot.py` (`make rescore`) applies `cap_wer(w) = min(w, 1.0)`
before computing anything (see `eval_common.cap_wer`). Re-running on the
same cache, same probe checkpoint, same test split:

| Trigger | Uncapped AVR (EXP-01, contaminated) | Capped AVR (EXP-00, rescore) |
|---|---|---|
| Oracle | 0.6539 | 0.0311 |
| Random | 0.0000 | 0.0000 |
| Confidence | **-1.3298** | **0.0147** |
| Argmax accent | 0.4543 | 0.0103 |
| Scalar probe (pilot) | n/a in EXP-01 | 0.0035 |

Capping doesn't just shrink the numbers — it **reverses the ranking**:
confidence goes from "worse than random" to the strongest non-oracle
trigger. A single WER = 223.0 utterance was enough to flip the sign of a
headline comparison. Kept here as a concrete, reproducible example of
aggregate-metric failure under a long-tailed error distribution, for
reference during the DDP/scaling work on this branch.

## Reproduce

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -c "
import json, glob
wers = [json.load(open(f))['wer'] for f in glob.glob('data/asr_cache/mlx_community_whisper_small_mlx/*.json')]
over = [w for w in wers if w > 1.0]
print(f'{len(over)}/{len(wers)} utterances have WER > 1.0')
print(f'max WER: {max(wers):.2f}, mean WER: {sum(wers)/len(wers):.3f}')
"
```
