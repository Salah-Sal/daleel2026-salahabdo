# Encoder multiseed campaign v2 (HEADROOM_AUDIT.md P5)

The kernel reads the private dataset named in `kernel/kernel-metadata.json`;
rebuild it with `../encoder-preflight/prepare_assets.py` (see `../README.md`).

Private T4 x2 kernel running twenty five-fold runs through two per-GPU
worker processes (`CUDA_VISIBLE_DEVICES=0/1`) over a shared queue:

| Block | Runs |
|---|---|
| Task 1 seeds | {quarter, mix, da} x seeds {20260710..20260714} = 15 |
| Task 2 target-segment (first real runs) | {quarter, mix, da} x 20260710 = 3 |
| Task 1 `--class-weighting sqrt` probe | {mix, da} x 20260710 = 2 |

Everything else is byte-identical to the committed campaign recipe, and
`--fold-seed` is never touched, so all runs share the local fold contract —
the quarter x 20260710 cell doubles as a GPU-fp16 vs local-CPU-fp32
replication anchor.

Export policy (documented deviation from campaign v1): Task 1 runs export
`oof_task_1_scores.jsonl` (paragraph_id + sigmoid floats + fold) and
`oof_decisions.jsonl` (paragraph_id + label codes) — no organizer text and
no offsets ever leave the ephemeral workspace; paragraph ids are public in
the task repo. These files feed the local seed-ensemble recomposition
(mean sigmoid -> route_task1_v2 gate). Task 2 exports safe artifacts only.

Guards: 13.5 GiB reserved-memory bar, strict determinism + eager fp16,
per-job 3 h timeout, 9.5 h launch deadline (12 h session cap), per-job
failure containment with a per-job status ledger in
`CAMPAIGN_COMPLETED.json` (`complete` only if all 20 validate).

```bash
.venv_kaggle/bin/kaggle kernels push -p shared-task/kaggle/encoder-multiseed-v2/kernel
.venv_kaggle/bin/kaggle kernels status salah1992/daleel-encoder-multiseed-v2
.venv_kaggle/bin/kaggle kernels output salah1992/daleel-encoder-multiseed-v2 \
  -p /private/tmp/daleel-encoder-multiseed-v2-output --force
```

## Run 1 incident (2026-07-14, kernel v1): 6/20 completed

Three defects, all fixed in kernel v2:

1. **Run-dir claim race** (10 failures, `ambiguous run dirs`): both workers
   share `experiments/`, and same-shape T1 jobs finish within seconds of each
   other, so the loser's before/after set diff contained the sibling's
   just-completed dir too. The failed jobs had *trained successfully* — the
   kernel refused to claim their output. v2 claims by unique job signature:
   task (from the trainer dir name) + (model repository, seed,
   class_weighting) read from each candidate's `config.json`.
2. **Stale bundle** (8 failures, instant `exit 1`): dataset v1 was built
   before `camelbert-da` entered `ENCODER_SPECS` (2026-07-11), so every da
   job died on registry lookup. Bundle re-staged from the current repo and
   pushed as dataset version 2.
3. **stdout-only error capture**: tracebacks go to stderr, so failures
   surfaced as `exit 1: ` with no message. v2 records both tails.

Run 1 scores kept for the fp16 determinism cross-check against run 2
(identical jobs must reproduce exactly): t1-quarter s10 0.5967 / s14 0.6209,
t1-mix s11 0.6145 / s13 0.6282, t1-mix-sqrt s10 0.6136, t2-quarter s10
0.6645 (first real Task 2 segment-encoder OOF).
