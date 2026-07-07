# AI Use Log

This document tracks how AI tools were used in this project.

## Tool

**Claude Code** (Anthropic) — CLI-based AI coding assistant, powered by Claude Opus 4.6.

## Usage Summary

The entire codebase was implemented with Claude Code over two sessions, guided by a detailed research plan written by the human researcher. The plan specified architecture, hyperparameters, evaluation methodology, and acceptance criteria for each phase. Claude Code generated all code, debugged runtime errors, and ran the pipeline.

## Detailed Log

| Phase | Task | AI Contribution | Human Contribution |
|-------|------|-----------------|-------------------|
| 0 | Environment setup | Generated `pyproject.toml`, resolved dependency issues | Specified dependency list and Python version constraints |
| 1 | Repo scaffold | Generated directory structure, configs, Makefile, docs | Designed project layout and config schema in research plan |
| 2 | Data pipeline | Wrote EdAcc loader and speaker-disjoint splits; debugged dataset ID (`edinburghcstr/edacc`), torchcodec API, and empty-split bug | Specified accent selection, subset size, split algorithm |
| 3 | ASR harness | Wrote mlx-whisper wrapper, WER computation, disk cache | Specified model pair, text normalization, caching strategy |
| 4 | Feature extraction | Wrote WavLM extractor with NaN guards | Specified layer extraction, pooling, memory constraints |
| 5 | Baselines | Implemented oracle, random, confidence, argmax-accent triggers and routing harness | Designed trigger protocol and operating curve methodology |
| 6 | Scalar probe | Implemented AccentednessProbe, training loop, calibration | Specified architecture, loss function, early stopping criteria |
| 7 | Evaluation | Implemented curves, per-accent slicing, speaker leakage analysis, plots | Designed evaluation metrics and diagnostic checks |
| 8 | Flywheel | Implemented drift detection (KS test) and hard-case mining | Specified simulation approach and mining criteria |
| 9-10 | Writeup & finalization | Drafted writeup and README with actual results; updated after pipeline runs | Directed what to include, reviewed figures |

## Bugs Found and Fixed by AI During Pipeline Execution

1. **EdAcc dataset ID**: `edinburgh-dawg/edacc` not found; searched HuggingFace API, found correct ID `edinburghcstr/edacc`
2. **torchcodec AudioDecoder API**: `datasets >= 3.x` returns AudioDecoder objects instead of dicts; wrote `_decode_audio()` handler for both formats
3. **jiwer v4 API**: `truth_transform` renamed to `reference_transform`; transforms require `ReduceToListOfListOfWords()`
4. **Empty test split**: Original loader sampled utterances in dataset order, getting only 2 speakers per accent; rewrote to round-robin across all speakers
5. **Split guarantees**: Original split logic could produce empty folds with few speakers; rewrote to guarantee >=1 speaker per fold
6. **JSON serialization**: numpy `float32` and `bool_` types not JSON-serializable; added explicit `float()` casts and custom encoder

### Round 2

| Phase | Task | AI Contribution | Human Contribution |
|-------|------|-----------------|-------------------|
| 0 | Re-scoring with capped WER | Implemented cap_wer, bootstrap CIs, paired bootstrap; discovered pilot evaluation was contaminated | Designed evaluation methodology and significance testing |
| Ext 1 | Gain-target probe | Retrained probes on escalation gain and capped WER targets | Specified target definitions and training protocol |
| Ext 2 | Learning curve + accent classifier | Implemented speaker-fraction learning curve and accent classifier; analyzed layer weights | Designed diagnostic experiments |
| Ext 3 | Hallucination baselines | Implemented no_speech_prob, compression_ratio, hallucination union triggers; confidence autopsy | Specified trigger designs and autopsy methodology |
| Ext 5 | Multi-task probe | Implemented lambda sweep, MI analysis, mutual information diagnostics | Specified multi-task loss formulation |

### Round 3

| Phase | Task | AI Contribution | Human Contribution |
|-------|------|-----------------|-------------------|
| 0 | Infrastructure | Extended eval_common.py (headroom_summary, combiner_eval, grid_eval); unit tests; config; Makefile | Designed plan with gate structure and decision rules |
| 1 | Headroom grid (EXP-09) | Implemented ASR grid decode, headroom grid eval, headroom sweep plot; decoded 3 new models (tiny, base, turbo) x 900 utterances; fixed stalled HuggingFace download | Specified model grid and evaluation methodology |
| 2 | Composite trigger (EXP-10) | Implemented acoustic features extraction, composite eval; diagnosed and fixed NaN contamination from tiny model's ASR cache | Specified combiner architecture and GBM fallback |
| 3 | Temporal std (EXP-11) | Implemented WavLM stats extraction, temporal eval with subset analysis; applied NaN fix from Phase 2 | Specified temporal variability feature and subset analysis |
| 5 | Synthesis | Generated COMPARISON.md (66 entries), writeup-v3.md, updated docs | Directed narrative structure and interpretation |

## What AI Did NOT Do

- The research plan (problem definition, architecture choices, evaluation methodology, accent selection rationale) was written by the human researcher before any code was generated
- All decisions about what constitutes a meaningful result vs. a failure were made by the human
- Interpretation of results (e.g., why confidence baseline fails, why routing benefit is concentrated in 2-3 accents) was discussed collaboratively
- The human directed pipeline execution step by step and decided when results were satisfactory
