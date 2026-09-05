# Experiment log

One folder per run (`YYYYMMDD-HHMMSS-...Z-<name>-<hash>/`) or per campaign
(`YYYYMMDD-<name>/`) containing

- `config.json` — the exact resolved configuration, enough to reproduce;
- `metrics.json` — scores from `daleel.metrics` (official-scorer ports);
- from the evening of 2026-07-10 onward, `provenance.json` (models, data
  hashes, fold membership, code fingerprint) and `COMPLETED.json` (sha256
  receipt of the run's files, checked by `daleel.artifacts`);
- `REPORT.md` in campaign folders — what was tried, what happened, the verdict.

Checkpoints and prediction files inside experiment folders are gitignored;
configs, metrics, provenance, receipts, and reports are committed. Keep the
index current:

| Folder | Task | Track | Setting | Model | Dev macro-F1 | One-line takeaway |
|---|---|---|---|---|---|---|
| `20260709-d10-bakeoff/` | 1+2 | closed | — | 14 models | local-val, not dev | **Bake-off report**: gemma-4-31b wins both tasks (T1 0.7024 / T2 0.654 local val); the `20260709-t{1,2}-zeroshot-*-val/` folders are its per-run inputs — regenerate the table with `scripts/bakeoff_table.py` |
| `20260709-d3-quote-vs-segment/` | 1+2 | closed | — | gemma-4-31b/26b | local-val, not dev | **D3 revision report**: quote-then-align adopted for Task 2 (0.6805 vs 0.6542, aligner oracle 0.999, 0 unaligned quotes); Task-1-via-extraction rejected (0.664 < 0.7024); inputs in `20260709-t{1,2}-quote-*-val/` |
| `20260709-t*-zeroshot-*-train8/` | 1+2 | closed | — | 3B/9B | smoke only | pipeline plumbing validation; llama-3.2-3b `:free` starved upstream |
| `20260709-d7-optimizers/` | 1+2 | closed | — | gemma-4-31b | local-val, not dev | **D7 optimizer report**: BFRS/MIPROv2/GEPA(×3 variants) on T1 and GEPA on T2 ALL rejected at the +0.02 paired bar — both tasks ship zero-shot seeds; incl. DeepSeek/Gemini/Cohere bake-off rows (gemma wins everything; free Gemini serving of the same checkpoint scores −0.048) |
| `20260710-v2-stage-architecture/` | 1+2 | closed | — | gemma-4-31b | local-val, not dev | **v2 architecture report**: error anatomy (labeling, not detection: T2 gold mass 0% unpredicted), oracle ceilings (0.86/0.88), zero-shot verify stages flat, decision-level judge compiles lift decision accuracy (GEPA AN 0.74→0.88) and optval macro (+0.033) but FAIL frozen-val transfer (+0.001, winner's curse) — champions unchanged |
| `20260710-233257-145267Z-sparse-tboth-both-legacy-train-723ea1b146/` | 1+2 | closed | both | TF-IDF + LinearSVC | OOF 0.5728 T1 cross-fit / 0.5832 T2; fixed-182 confirm 0.6702 / 0.6232 | **Canonical strict no-LM CPU run**: complete provenance + raw margins; T1 same-OOF threshold diagnostic is 0.6176; exact seeded rerun hashes match |
| `20260710-231400-297323Z-encoder-t1-camelbert-msa-both-legacy-train-smoke-fe95497542/` + `20260710-231411-214841Z-encoder-t2-camelbert-msa-both-legacy-train-smoke-897186f054/` | 1+2 | closed | both | CAMeLBERT-MSA | smoke only | **Non-generative encoder plumbing**: both real-data paths complete through backward pass, OOF decode, official scorer, provenance, and completion marker; deliberately non-comparable one-step frozen-head runs |
| `20260711-v3-proposal-atom-role/` | 1+2 | closed | both | gemma-4-31b | frozen-182 T2 0.6995 (adopted) / T1 champion retained | **v3 campaign report**: proposal→atom→role passes the +0.02 OOF gate on both tasks (T2 +0.0387, T1 fusion +0.0436), frozen confirmation adopts **Task 2 0.6995 vs 0.6805**; T1 fusion fails transfer (0.6745 vs 0.7024) — first pre-registered win; incl. champion run-to-run stability finding (±0.03–0.04 macro on identical paragraphs) |

The v3 campaign writes microsecond/config-hash run directories and is indexed
by its hash-bound artifacts rather than a pre-created folder. Its release
contract and exact command sequence were fixed in the v3 milestone note. A run is valid
only when `COMPLETED.json` is present; interrupted directories are never inputs
to aggregation or deployment.
