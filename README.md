# SalahAbdo at Daleel 2026

System, experiment records, and preregistration ledger for the SalahAbdo entry to the
[Daleel 2026 shared task](https://github.com/Argmining/Daleel2026) on Arabic argumentative
discourse mining (ArabicNLP 2026, co-located with EMNLP 2026, Budapest). Closed track,
`both` genres: **first of ten** teams on Task 1 (which argumentative-unit types a paragraph
contains; macro-F1 0.712) and **fifth of eight** on Task 2 (where their spans are;
span-F1 0.7316).

Paper: Salah Abdo, *SalahAbdo at Daleel 2026: Routing Each Label to LLM Votes, a
Fine-Tuned Encoder, or Sequence Structure*, ArabicNLP 2026 (to appear). `shared-task/paper/` holds the
submitted version (tex, bib, ACL style files, PDF; revision of 2026-08-14). The camera-ready
version will replace it and be attached to a GitHub release; `CITATION.cff` carries the citation.

## What is in this repository

| Path | Content |
|---|---|
| `shared-task/src/daleel/` | the `daleel` package: data loading and frozen splits, ports of the official scorers, label policy and verbatim prompts, DSPy programs and optimizer metrics, encoder and sparse baselines, provenance and completion receipts, submission packaging |
| `shared-task/scripts/` | 47 command-line entry points: model bake-off, DSPy compiles, per-label routing, verifiers, registered gates, deployment twins, paper analyses |
| `shared-task/tests/` | 244 tests; 28 of them read the organizers' clone (18 fail and 10 skip without it) |
| `shared-task/experiments/` | 239 run and campaign directories, indexed in `experiments/README.md`: 233 hold `config.json` and 219 `metrics.json`; the 72 runs from the evening of 2026-07-10 onward also hold `provenance.json` and a `COMPLETED.json` receipt (71); 5 campaign folders hold a `REPORT.md` |
| `shared-task/kaggle/` | the Kaggle kernels used for GPU encoder training |
| `shared-task/paper/` | the paper: tex, bib, ACL style files, PDF (submitted version until the camera-ready replaces it) |
| `shared-task/CREATIVE_HEADROOM_RESEARCH.md` | the preregistration ledger, with its commit history: each gate registered before the corresponding fit, each verdict appended after |
| `shared-task/HEADROOM_AUDIT.md`, `TIER1_ROUTING_MILESTONE.md`, `STRUCTURAL_FOREST_EXPERIMENT.md` | the three registered designs that the ledger and the code cite |
| `COMMIT_MAP.md` | development commit to public commit, so the hashes cited in the paper and in the ledger resolve here |
| `REPRODUCING.md` | what can be verified from the records, what can be re-run, and what cannot be reproduced from this repository; where each paper number lives |
| `shared-task/EVAL_WEEK_PROBES.md` | the post-freeze evaluation-week probes at rule level, with their official scores; the two scored leaderboard entries come from here |
| `shared-task/official_scores/` | Codabench readouts of every scored submission of this entry (development phase, frozen test recipes, probes) and the organizers' final standings for it |

Not distributed: prediction files, compiled prompt states, GEPA logs, and checkpoints inside
run directories (they contain dataset text or are large); the organizers' data (clone it,
see below); and three 2026-07-09 bake-off `metrics.json` files whose error samples embedded
dataset paragraphs. Because the prediction files are absent, the composition scripts cannot be
replayed on the committed records; `REPRODUCING.md` says what can be verified, re-run, or not
reproduced.

## Setup

```bash
git clone https://github.com/Salah-Sal/daleel2026-salahabdo
cd daleel2026-salahabdo
git clone https://github.com/Argmining/Daleel2026 resources/repos/Daleel2026  # data and official scorer; the code loads from this path
git -C resources/repos/Daleel2026 checkout 49f000c                            # the commit the recorded runs read
cd shared-task
uv sync                 # DSPy 3.3.0b1, pandas, scikit-learn, pytest: enough for the LLM pipeline
uv sync --group train   # adds torch, transformers, safetensors; required for encoder work AND for the
                        # test suite (two test modules import torch and abort collection without it)
uv run pytest -q        # 244 tests, no network calls; 28 read the organizers' clone (18 fail, 10 skip without it)
```

LLM runs call OpenRouter and read `OPENROUTER_API_KEY` from a `.env` file at the repository
root (gitignored). `shared-task/README.md` has the layout, every entry point, and the
determinism and caching notes.

## How the records fit together

- Run directories from the evening of 2026-07-10 onward hold `config.json`, `metrics.json`,
  `provenance.json` (models, data hashes, fold membership, code fingerprint) and
  `COMPLETED.json`, a receipt listing the sha256 of the run's files.
  `daleel.artifacts.validate_completion_marker` checks the receipt before a run is used as
  input to anything else (here it raises on runs whose receipts list prediction files, which
  are not distributed; `REPRODUCING.md` has a check over the distributed files only). Earlier runs (the 2026-07-09 bake-off and the first campaigns)
  predate the receipt convention and hold `config.json` and `metrics.json`; campaign folders
  hold a `REPORT.md`.
- The ledger records each proposed change with its adoption bar before the run and the
  verdict after. The registry table in the paper cites those ledger commits;
  `COMMIT_MAP.md` translates them to this repository's history.

## Provenance notes

This repository is a curated release of a private development repository. Read these before
verifying anything by hash.

- The commit history covers the four registration documents only. Everything else entered
  in one commit from the development tree at its commit `00c0c41` (2026-09-04).
- Absolute paths of the development machine were replaced by `<repo>/` in experiment
  records. The `COMPLETED.json` receipts were recomputed after that substitution, so every
  receipt entry for a distributed file verifies; entries for files that are not distributed
  are as originally written.
- `git_commit` and `python_tree_sha256` in `provenance.json` refer to the private
  development tree and do not resolve here.
- Comments in the code and in reports cite internal design notes (DSPy design space, metric
  design, guideline-native redesign, model-config optimization, literature-review notes, v3
  milestone, encoder and sparse baseline notes). Those notes are not part of this release.
- The label definitions and the demonstration sentences in `src/daleel/guideline_demos.py`
  and `src/daleel/guideline_policy.py` are quoted from the organizers' annotation guidelines
  (`Annotation Guidelines EN.md` in their repository).

## License

MIT for everything in this repository (`LICENSE`). The paper in `shared-task/paper/` is under the ACL
Anthology's CC BY 4.0. The organizers' data and guidelines are theirs; consult their
repository for terms.
