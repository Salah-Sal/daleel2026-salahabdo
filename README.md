# SalahAbdo at Daleel 2026

System, experiment records, and preregistration ledger for the SalahAbdo entry to the
[Daleel 2026 shared task](https://github.com/Argmining/Daleel2026) on Arabic argumentative
discourse mining (ArabicNLP 2026, co-located with EMNLP 2026, Budapest). Closed track,
`both` genres: **first of ten** teams on Task 1 (which argumentative-unit types a paragraph
contains; macro-F1 0.712) and **fifth of eight** on Task 2 (where their spans are;
span-F1 0.7316).

Paper: Salah Abdo, *SalahAbdo at Daleel 2026: Routing Each Label to LLM Votes, a
Fine-Tuned Encoder, or Sequence Structure*, ArabicNLP 2026 (to appear). The camera-ready
PDF and BibTeX will be added under `shared-task/paper/` and attached to a GitHub release;
`CITATION.cff` carries the citation.

## What is in this repository

| Path | Content |
|---|---|
| `shared-task/src/daleel/` | the `daleel` package: data loading and frozen splits, ports of the official scorers, label policy and verbatim prompts, DSPy programs and optimizer metrics, encoder and sparse baselines, provenance and completion receipts, submission packaging |
| `shared-task/scripts/` | 47 command-line entry points: model bake-off, DSPy compiles, per-label routing, verifiers, registered gates, deployment twins, paper analyses |
| `shared-task/tests/` | 244 tests |
| `shared-task/experiments/` | 239 run directories (config, metrics, provenance, completion receipt, reports), indexed in `experiments/README.md` |
| `shared-task/kaggle/` | the Kaggle kernels used for GPU encoder training |
| `shared-task/CREATIVE_HEADROOM_RESEARCH.md` | the preregistration ledger, with its commit history: each gate registered before the corresponding fit, each verdict appended after |
| `shared-task/HEADROOM_AUDIT.md`, `TIER1_ROUTING_MILESTONE.md`, `STRUCTURAL_FOREST_EXPERIMENT.md` | the three registered designs that the ledger and the code cite |
| `COMMIT_MAP.md` | development commit to public commit, so the hashes cited in the paper and in the ledger resolve here |

Not distributed: prediction files, compiled prompt states, GEPA logs, and checkpoints inside
run directories (they contain dataset text or are large); the organizers' data (clone it,
see below); and three 2026-07-09 bake-off `metrics.json` files whose error samples embedded
dataset paragraphs.

## Setup

```bash
git clone https://github.com/Salah-Sal/daleel2026-salahabdo
cd daleel2026-salahabdo
git clone https://github.com/Argmining/Daleel2026 resources/repos/Daleel2026  # data and official scorer; the code loads from this path
cd shared-task
uv sync                 # DSPy 3.3.0b1, pandas, scikit-learn, pytest
uv sync --group train   # adds torch, transformers, safetensors for encoder work
uv run pytest -q        # 244 tests, no network; 18 of them read the organizers' clone
```

LLM runs call OpenRouter and read `OPENROUTER_API_KEY` from a `.env` file at the repository
root (gitignored). `shared-task/README.md` has the layout, every entry point, and the
determinism and caching notes.

## How the records fit together

- Each run directory holds `config.json`, `metrics.json`, `provenance.json` (models, data
  hashes, fold membership, code fingerprint) and `COMPLETED.json`, a receipt listing the
  sha256 of the run's files. `daleel.artifacts.validate_completion_marker` checks the
  receipt before a run is used as input to anything else.
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

MIT for everything in this repository (`LICENSE`). The paper, once added, is under the ACL
Anthology's CC BY 4.0. The organizers' data and guidelines are theirs; consult their
repository for terms.
