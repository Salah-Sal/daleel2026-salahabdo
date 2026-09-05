# shared-task

The system, its records, and the ledger. Everything runs from this directory with `uv`.

## Layout

```
shared-task/
├── src/daleel/                      the package (module map below)
├── scripts/                         command-line entry points (list below)
├── tests/                           244 tests: uv run pytest -q
├── experiments/                     one directory per run; experiments/README.md is the index
├── kaggle/                          Kaggle kernels for GPU encoder training (kaggle/README.md)
├── CREATIVE_HEADROOM_RESEARCH.md    preregistration ledger: gates and verdicts in commit order
├── HEADROOM_AUDIT.md                measured-lever audit and the P0-P7 verdicts (2026-07-14)
├── TIER1_ROUTING_MILESTONE.md       registered design of the routed Task 1 system (2026-07-11)
├── STRUCTURAL_FOREST_EXPERIMENT.md  registered design of the discourse-forest experiment (2026-07-14)
├── pyproject.toml, uv.lock          dspy pinned to 3.3.0b1; the `train` group adds torch/transformers
└── data/, configs/, notebooks/      empty placeholders; the data lives in ../resources/repos/Daleel2026
```

## Quickstart

```bash
uv sync                  # base environment
uv sync --group train    # encoder work (torch, transformers, safetensors)
uv run pytest -q         # 244 tests; 18 read ../resources/repos/Daleel2026
```

`daleel.data` loads the organizers' files straight from `../resources/repos/Daleel2026`:

```python
from daleel import load_records, task1_labels, TRAIN_TASK1
gold = task1_labels(load_records(TRAIN_TASK1, genre="editorial"))
```

## Keys, determinism, caching

- Create `.env` at the repository root (gitignored): `OPENROUTER_API_KEY` for every
  `gemma-4-31b-paid` run; `DEEPSEEK_API_KEY` only for GEPA with
  `--reflection-model deepseek-v4-pro`; `GEMINI_API_KEY` and `COHERE_API_KEY` only for those
  bake-off rows. `daleel.runtime` loads the file before any DSPy import.
- All runs use temperature 0. Splits are frozen in `daleel.splits` (seed 20260709: 430
  train / 182 validation; optimizer-internal re-split seed 20260710: 281 / 149, so optimizers
  never see the frozen validation set).
- DSPy's disk cache lives at `experiments/.dspy_cache/` (gitignored); re-running a command
  returns cached completions. Provider-side nondeterminism exists, so expect exact numbers
  from the committed `metrics.json` files rather than from fresh API calls.
- `DALEEL_MLFLOW=0` disables MLflow logging (the test suite sets it).

## Module map (`src/daleel/`)

| Module | Purpose (first docstring line) |
|---|---|
| `align.py` | Quote-to-offset alignment for quote-then-classify span prediction (D3). |
| `artifacts.py` | Hash and validate model-generated artifact lineage across Daleel stages. |
| `candidates.py` | Deterministic candidate atomization for the decomposed Daleel program. |
| `constants.py` | Task-level constants for Daleel 2026. |
| `cv.py` | Strict compatibility and fold-membership checks for v3 CV runs. |
| `data.py` | Loaders for the official Daleel2026 dataset. |
| `discourse_forest.py` | Validated, label-blind discourse forests over proposal-derived atoms. |
| `dspy_discourse.py` | DSPy parser for label-blind discourse forests over frozen atoms. |
| `dspy_metrics.py` | DSPy optimization metrics for both tasks (design item D8). |
| `dspy_programs.py` | DSPy programs for both tasks (design axes D2, D3, D4). |
| `dspy_span_roles.py` | DSPy v3: proposal atomization plus independently optimized role decisions. |
| `encoder_baseline.py` | Non-generative Arabic encoder baselines for Daleel Tasks 1 and 2. |
| `folds.py` | Deterministic paragraph-grouped folds for Daleel model selection. |
| `guideline_demos.py` | The official guideline examples as DSPy demos for the guideline-native signatures. |
| `guideline_policy.py` | Guideline-native label policy: seeded from the OFFICIAL annotation guidelines. |
| `guideline_programs.py` | Guideline-native two-stage program. |
| `io.py` | JSONL reading/writing with the encoding conventions this task needs. |
| `metrics.py` | Evaluation metrics for the two Daleel 2026 subtasks. |
| `models.py` | Candidate model ladder and LM construction (D10/D15). |
| `policy.py` | Canonical label policy: one phrasing for definitions and corpus rules. |
| `provenance.py` | Compliance checks and reproducibility manifests for Daleel runs. |
| `role_artifacts.py` | Complete, hash-bound artifacts for the v3 atom-role classifier. |
| `runtime.py` | Process setup that must happen BEFORE `import dspy`. |
| `segment.py` | Deterministic Arabic segmenter for Task 2 (D3: segment-then-classify). |
| `sparse_baseline.py` | Strict no-language-model TF-IDF/LinearSVC baselines for Daleel. |
| `splits.py` | D9: gold-cleaning policy and the fixed local train/val split. |
| `submission.py` | Validation and packaging of Codabench submissions. |

## Entry points (`scripts/`)

| Script | Purpose (first docstring line) |
|---|---|
| `aggregate_role_cv.py` | Aggregate strict v3 outer folds and execute the fixed adoption gate. |
| `analyze_decomposition.py` | Measure the proposal/atomization oracle before spending optimizer calls. |
| `audit_signal_variables.py` | Ceiling audit for the (source, form, stance) signal-variable design. |
| `bakeoff_table.py` | Tabulate the D10 bake-off from experiments/*/metrics.json. |
| `co_judge_rank.py` | Many-shot CO ranking judge (HEADROOM_AUDIT.md P1). |
| `co_specialist_rank.py` | CO contrastive many-shot specialist as a 4th rank feature (registered gate in CREATIVE_HEADROOM_RESEARCH.md, frozen 2026-07-15 before any call). |
| `compare_structural_cv.py` | Compare complete baseline/flat/forest/shuffled OOF campaigns. |
| `compile_judge.py` | Decision-level compile of the Task 1 judge (v2 propose-then-verify). |
| `compile_program.py` | Optimizer compile runner: D7 stages 1-3 (BFRS, MIPROv2, GEPA). |
| `compile_span_roles.py` | Compile and officially rerank the v3 one-atom Daleel role program. |
| `deploy_encoder_task1.py` | Deploy the Task 1 encoder: train on all 430 train-side paragraphs, predict an external input. |
| `deploy_encoder_task2.py` | Deploy the Task 2 segment encoder and score an external input. |
| `ensemble_encoder_seeds.py` | Mean-sigmoid seed ensemble for Task 1 encoder OOF runs (HEADROOM_AUDIT.md P5). |
| `ensemble_task2_segment_scores.py` | Average compatible Task 2 segment-encoder score runs. |
| `freeze_discourse_forests.py` | Freeze label-blind discourse forests over quote-proposal atoms. |
| `judge_stage_test.py` | Isolated test of the Task 1 judge stage on frozen proposals (v2 design). |
| `metric_alignment_check.py` | Pre-flight alignment study for D8 (metric design): which per-example proxy best RANKS predictors the way the official corpus-level scorers do? |
| `package_submission.py` | Validate a predictions jsonl and package it as a Codabench submission. |
| `paper_analysis_bank.py` | Paper analysis bank: A1 threshold audit, A2 vote-scaling curves, A3 span confusion + coarse-family rebound (literature-review analyses, zero-risk). |
| `relabel_stage_test.py` | Isolated test of the Task 2 relabel stage on frozen extractions (v2 design). |
| `route_task1.py` | Tier-1 Task 1 router: per-label routing + self-consistency + CO budget. |
| `route_task1_deploy.py` | Compose the adopted tier-1 routed Task 1 system on an external input. |
| `route_task1_v2.py` | Tier-1 Task 1 router v2: multi-source consensus legs + CO rank fusion. |
| `route_task1_v2_deploy.py` | Compose the adopted tier-1 v2 routed Task 1 system on an external input. |
| `run_decomposed.py` | Run the release-safe v3 proposal -> atom -> role pipeline. |
| `run_zero_shot.py` | End-to-end zero-shot runner: program -> predictions -> official scores. |
| `select_task1_fusion.py` | Cross-fit and adoption-gate Task 1 direct/span reconciliation. |
| `st_judge_seed_deploy.py` | Deploy twin of the v2.1 ST seed judge (judge_stage_test.py, ST drop-only leg). |
| `st_verifier_v2.py` | ST contrastive verifier v2 (registered gate in CREATIVE_HEADROOM_RESEARCH.md). |
| `st_verifier_v2_deploy.py` | Deploy twin of the ST contrastive verifier v2 (adopted 2026-07-15). |
| `t1_bundle_v3_gate.py` | T1 bundle-v3 gate (registered, CREATIVE_HEADROOM_RESEARCH.md commit 56d6f30). |
| `t1_tapt_gate.py` | T1 TAPT gate (registered, CREATIVE_HEADROOM_RESEARCH.md commit cd77061). |
| `t2_an_rescue.py` | T2 AN-rescue span judge (registered gate in CREATIVE_HEADROOM_RESEARCH.md). |
| `t2_constrained_relabel.py` | Constrained relabel of Task 2 spans that contradict the routed Task 1 set (HEADROOM_AUDIT.md P3). |
| `t2_dev_stacker.py` | Cross-fit a dev-supervised Task 2 span relabel/drop stacker and deploy it. |
| `t2_eval_week_filter.py` | Audit and deploy eval-week Task 2 span-presence filters. |
| `t2_g1_tau_gate.py` | T2 "G1" calibration-temperature gate (registered, CREATIVE_HEADROOM_RESEARCH.md 0f6b709). |
| `t2_paired_bootstrap.py` | Paired paragraph bootstrap for Task 2 prediction comparisons. |
| `t2_presence_rescue.py` | Add conservative Task 2 spans under a Task 1 presence constraint. |
| `t2_rescue_crossfit.py` | Cross-fit the two-stage Task 2 encoder rescue on released dev references. |
| `t2_s2_combiner_gate.py` | T2 "S2" confidence-selective span combiner (registered gate, CREATIVE_HEADROOM_RESEARCH.md). |
| `t2_structural_gate.py` | T2 structural re-decode bundle "S1" (registered gate, CREATIVE_HEADROOM_RESEARCH.md). |
| `t2_structural_gate_deploy.py` | S1 structural bundle — deploy twin (CREATIVE_HEADROOM_RESEARCH.md, adopted gate). |
| `t2_vote_pilot.py` | Task 2 atom-vote self-consistency pilot (HEADROOM_AUDIT.md P2). |
| `tapt_pretrain.py` | T1 TAPT: continued MLM pretraining on organizer-provided input text. |
| `train_encoder_baseline.py` | Train and evaluate DSPy-free Arabic encoder baselines with strict OOF CV. |
| `train_sparse_baseline.py` | Run strict TF-IDF/LinearSVC Daleel baselines with OOF evaluation. |
