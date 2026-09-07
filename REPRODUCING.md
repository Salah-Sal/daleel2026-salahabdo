# Reproducing the paper from this repository

This file is for a reader who wants to check or re-run the numbers in the paper. It
separates three things: what can be verified from the committed records without any model
call, what can be re-run deterministically, and what cannot be reproduced from this
repository at all. Read it before the setup block in `README.md`.

## 1. Inputs

- **Organizers' data.** Clone https://github.com/Argmining/Daleel2026 into
  `resources/repos/Daleel2026` at the repository root; the code resolves that path from its
  own location, so commands work from `shared-task/`. The recorded runs read the clone at
  commit `49f000c`: `data/train/train_task_{1,2}.jsonl` (612 paragraphs; the code splits
  them 430/182 with seed 20260709), `data/dev/dev_in.jsonl` with
  `dev_task_{1,2}_ref.jsonl` (217 paragraphs; the references were released during the
  evaluation phase), `data/test/test_in.jsonl` (213 paragraphs; no test references have been
  released), and the annotation guidelines. Every `provenance.json` records the sha256 of the
  organizer files a run read (`data_artifacts`, `input.sha256`); the test input hashes to
  `65a95069ef68584ddb7720150de7be5797bdbadb1e507a2ec68ce07cd336e6ef`.
- **Environment.** `uv sync --group train` inside `shared-task/` (Python 3.11 or later;
  `uv.lock` pins everything, DSPy 3.3.0b1). LLM stages read `OPENROUTER_API_KEY` from
  `.env` at the repository root. `DALEEL_MLFLOW=0` turns off MLflow logging.
- **Models.** `gemma-4-31b-it` through OpenRouter (`openrouter/google/gemma-4-31b-it`; the
  provider pin used in the CFG-J1 probe is in `src/daleel/models.py`), `qwen3-32b` and
  `llama-3.3-70b` as verifier co-signals, and CAMeLBERT-MSA-quarter at Hugging Face revision
  `3e48534705c153737cbec1c5748bb02359b7b239` (recorded in every encoder `config.json`).
  OpenRouter can retire or re-route a model; each run records the served endpoint in
  `provenance.json`.

## 2. Tier 1: verify from the records (no API, no GPU)

- **Test suite.** `uv run pytest -q` from `shared-task/`: 244 tests, including parity
  tests that import the organizers' own scoring scripts and compare them with the ports in
  `daleel.metrics`.
- **Every gate and score in the registry.** `shared-task/experiments/README.md` maps each
  row of the paper's registry table to the run directories that hold its `metrics.json`,
  and lists the deployment twins for the development and test phases. The numbers in the
  paper are read from those files, not recomputed.
- **Receipts.** `COMPLETED.json` in each run from the evening of 2026-07-10 onward lists the
  sha256 of the run's files. Entries for prediction files refer to files that are not
  distributed, so `daleel.artifacts.validate_completion_marker` raises on most runs here.
  To check the distributed files only:

  ```bash
  uv run python - <<'EOF'
  import hashlib, json, pathlib
  run = pathlib.Path("experiments/20260727-044517-888390Z-v3-decomposed-gemma-4-31b-paid-both-test_in-aada058d66")
  for row in json.load(open(run / "COMPLETED.json"))["files"]:
      p = run / row["relative_path"]
      if p.is_file():
          print("ok " if hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"] else "BAD", row["relative_path"])
      else:
          print("absent (not distributed)", row["relative_path"])
  EOF
  ```

- **Bake-off table.** `uv run scripts/bakeoff_table.py` rebuilds the model bake-off from
  the committed `metrics.json` files.
- **Appendix tables.** The threshold audit, the vote-scaling model, and the confusion-mass
  figures come from `experiments/20260716-133715-667411Z-paper-analysis-bank-n430-d7e5a011ed/metrics.json`
  (`a1_lipton_threshold_audit`, `a2_vote_scaling`, `a3_confusion_families`).
- **Encoder variants.** The per-seed out-of-fold score files under
  `kaggle/encoder-multiseed-v2/results/` are the only prediction-level artifacts
  distributed (paragraph ids and sigmoid scores, no text). `scripts/ensemble_encoder_seeds.py --runs <dir> ...`
  accepts those directories directly; the five-seed means quoted in the paper are the means
  of the `metrics.json` macro-F1 across seeds.
- **Registration order.** `git log --follow shared-task/CREATIVE_HEADROOM_RESEARCH.md`
  shows each registration and verdict commit with its date; `COMMIT_MAP.md` translates the
  development hashes cited in the paper.
- **Official scores.** `shared-task/official_scores/` holds the Codabench readouts for
  every scored development-phase and evaluation-phase submission of this entry, and the
  organizers' final standings for it.

## 3. Tier 2: re-run the deterministic parts (CPU or Kaggle GPU, no API)

- **Sparse baseline.** `uv run scripts/train_sparse_baseline.py --task both`
  reproduces the strict no-language-model run in seconds; the run's `config.json`
  (`arguments`) has the exact settings.
- **Encoder out-of-fold runs.** `uv run scripts/train_encoder_baseline.py --task 1 --model camelbert-msa-quarter --epochs 4 --folds 5`
  (Task 2: `--task 2`). Every recorded encoder run stores its full argument set under
  `arguments` in `config.json`, including `seed` and `fold_seed`. The CPU fp32 recipe took
  about 23 minutes for a Task 1 five-fold run and about 14 minutes for a deployment fit on
  all 430 paragraphs on the development machine. The paper reports digit-for-digit
  reproduction across duplicated fp16 GPU jobs; a different CPU or BLAS build may move the
  last digits.
- **Encoder deployments.** `scripts/deploy_encoder_task1.py` and
  `scripts/deploy_encoder_task2.py` train on all 430 paragraphs and score an external
  input; they assert their hyperparameters against the out-of-fold run they are given.
- **Kaggle campaigns.** The kernels under `kaggle/` read a private Kaggle dataset that
  contains organizer text and cannot be shared. Rebuild it under your own account with
  `kaggle/encoder-preflight/prepare_assets.py` and point `dataset_sources` in each
  `kernel-metadata.json` at it.

## 4. Tier 3: re-run the LLM stages (OpenRouter, paid, noisy)

Single zero-shot runs on 182 or 217 paragraphs move by 0.03 to 0.04 macro-F1 between
identical calls to the same endpoint; composed out-of-fold scores move less. Expect to
land inside that band, not on the recorded digits. The recorded stage order for the pinned
test-phase recipes is:

1. Task 1 zero-shot rollouts: `scripts/run_zero_shot.py --task 1 --model gemma-4-31b-paid --temperature 0.7 --rollout-id r0`
   (through `r4`), five rollouts on the 430 training paragraphs and five on the target input
   (`--input` and `--source-id` select the organizer file).
2. Quote proposals and role decisions: `scripts/run_decomposed.py` needs a compiled role
   state; recompile it with `scripts/compile_span_roles.py` (the recorded compile took
   about 28 minutes and is part of the campaign the paper prices at about US$13.5).
3. Encoder out-of-fold run and deployment for each task (Tier 2).
4. Task 1 composition: `scripts/route_task1_v2_deploy.py` with `--ot-source encoder`,
   then `scripts/judge_stage_test.py --drop-labels ST` and `scripts/st_verifier_v2_deploy.py`
   with one `qwen3-32b` and one `llama-3-3-70b-paid` zero-shot run as co-signals.
5. Task 2 composition: `scripts/t2_structural_gate_deploy.py` over the role-decision run,
   the quote run, and the Task 2 encoder deployment.

The exact argument lists of every stage as it ran on the test input are in the
`config.json` files of the `experiments/20260727-*` directories; the upstream run
names are in `provenance.json` (`upstream_artifacts`) for the LLM and encoder stages,
and in the `config.json` argv for the router, judge, verifier and structural-decode
stages, which write no provenance file or completion receipt. The paper prices the whole
test-phase pass at roughly 3.9k calls and US$10.

## 5. What cannot be reproduced from this repository

- **The two scored leaderboard entries** (Task 1 0.712, Task 2 0.7316) are evaluation-week
  probe compositions. They were assembled interactively from cached prediction files and
  have no run directory and no script; `shared-task/EVAL_WEEK_PROBES.md` states each
  probe's rules and official score. The Task 2 span filters are implemented in
  `scripts/t2_eval_week_filter.py`; the Task 1 leg swaps are described at rule level only.
- **Prediction files, compiled prompt states, GEPA logs, and the DSPy cache** are not
  distributed because they embed dataset text. Every composition script (the routers, the
  structural re-decode, the registered gates, `paper_analysis_bank.py`,
  `t2_s1_ablation.py`) reads such files from earlier runs, so none of them can be replayed
  on the committed records; they need fresh upstream runs first. Two gate scripts
  (`t2_g1_tau_gate.py`, `t2_s2_combiner_gate.py`) read prediction files at import time and
  fail even on `--help` without them.
- **The deployed Task 2 role state** (`compiled/role.json` of the run ending in
  `bc5f4625f5`, a GEPA compile self-reflected with `gemma-4-31b-it`) is one of those
  compiled states. Its selected candidate is number 0, the unmodified seed with no
  demonstrations, so the deployed instruction text is the `ClassifyOneAtom` docstring in
  `src/daleel/dspy_span_roles.py` and is in the release; the five outer-fold compiles
  selected rewritten instructions, which are not. What the deployment also needs and
  the release withholds is the role bundle's demonstration-selector memory
  (`role_bundle.json`, `role_memory.jsonl`), which holds training-paragraph text. The
  verbatim zero-shot prompts are in `src/daleel/policy.py`, `guideline_policy.py`,
  `dspy_programs.py`, and `dspy_span_roles.py`.
- **Test scores.** No test references exist outside Codabench, so nothing on the 213 test
  paragraphs can be scored locally.
- **Development-tree hashes.** `git_commit` and `python_tree_sha256` in `provenance.json`
  refer to the private development repository.

## 6. Where each paper number lives

| Paper item | Where |
|---|---|
| Registry table rows 1 to 29 | `shared-task/experiments/README.md`, section "Registry rows and deployments" |
| Progression table, development rows | `shared-task/official_scores/task_{1,2}_dev_history.json` |
| Progression table, frozen-recipe test rows | `shared-task/official_scores/task_{1,2}_eval.json` |
| Progression table, final probes and ranks | `shared-task/official_scores/final_standings.json`, `shared-task/EVAL_WEEK_PROBES.md` |
| Transfer ledger | predicted values in the gate runs' `metrics.json`; observed values in `official_scores/` |
| Threshold audit, vote model, confusion mass | the paper-analysis-bank run named above |
| S1 leave-one-out ablation (Appendix C) | `shared-task/experiments/20260907-160206-t2-s1-ablation-loo-n430/metrics.json`; script `scripts/t2_s1_ablation.py`. Zero API, but it reads the role runs' prediction files, so it is inspectable here and re-runnable only after fresh upstream runs. Its `full` row reproduces the recorded 0.7206 and its `v3` baseline 0.6934, which is the run's own control check. `experiments/README.md` explains how to read the `-Viterbi` row (it collapses to v3 + Rule-A) and why its re-fitted lambda is a plateau, not a finding. |
| Bake-off | `experiments/20260709-d10-bakeoff/REPORT.md`, `scripts/bakeoff_table.py` |
| Encoder-variant means | `kaggle/encoder-multiseed-v2/results/*/metrics.json`, `experiments/20260714-2345*-t1-encoder-ensemble-*-5seed-fp16-n430-*` |
| Paid-API ledger and GPU hours | not verifiable from the repository |
