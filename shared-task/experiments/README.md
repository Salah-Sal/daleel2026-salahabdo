# Experiment log

One folder per run (`YYYYMMDD-HHMMSS-...Z-<name>-<hash>/`) or per campaign
(`YYYYMMDD-<name>/`) containing

- `config.json` — the exact resolved configuration, enough to reproduce;
- `metrics.json` — scores from `daleel.metrics` (official-scorer ports);
- from the evening of 2026-07-10 onward, for LLM, encoder and sparse-baseline
  runs, `provenance.json` (models, data hashes, fold membership, code
  fingerprint) and `COMPLETED.json` (sha256 receipt of the run's files, checked
  by `daleel.artifacts`); the composition, gate and judge scripts
  (`route_task1*.py`, `t2_structural_gate*.py`, `st_verifier_v2*.py`,
  `st_judge_seed_deploy.py`, `judge_stage_test.py`, and the registered gate
  scripts of rows 22 to 29) write `config.json` and `metrics.json` only, with
  their exact argv, upstream run names and, for Task 1, input sha256s inside
  them;
- `REPORT.md` in campaign folders — what was tried, what happened, the verdict.

Checkpoints and prediction files inside experiment folders are gitignored;
configs, metrics, provenance, receipts, and reports are committed.

Timestamps: `daleel.artifacts.create_experiment_dir` stamps folder names in UTC
with a `Z` suffix; folders without the suffix (the composition and gate scripts)
are stamped in the machine's local time (America/New_York, UTC−4 in July) when
the script finishes, so they mark the end of the computation; git commit dates
carry their `-04:00` offset. Convert to one zone before comparing a folder to a
commit. Keep the index current:

| Folder | Task | Track | Setting | Model | Dev macro-F1 | One-line takeaway |
|---|---|---|---|---|---|---|
| `20260709-d10-bakeoff/` | 1+2 | closed | — | 14 models | local-val, not dev | **Bake-off report**: gemma-4-31b wins both tasks (T1 0.7024 / T2 0.654 local val); the `20260709-t{1,2}-zeroshot-*-val/` folders are its per-run inputs — regenerate the table with `scripts/bakeoff_table.py` |
| `20260709-d3-quote-vs-segment/` | 1+2 | closed | — | gemma-4-31b/26b | local-val, not dev | **D3 revision report**: quote-then-align adopted for Task 2 (0.6805 vs 0.6542, aligner oracle 0.999, 0 unaligned quotes); Task-1-via-extraction rejected (0.664 < 0.7024); inputs in `20260709-t{1,2}-quote-*-val/` |
| `20260709-t*-zeroshot-*-train8/` | 1+2 | closed | — | 3B/9B | smoke only | pipeline plumbing validation; llama-3.2-3b `:free` starved upstream |
| `20260709-d7-optimizers/` | 1+2 | closed | — | gemma-4-31b | local-val, not dev | **D7 optimizer report**: BFRS/MIPROv2/GEPA(×3 variants) on T1 and GEPA on T2 ALL rejected at the +0.02 paired bar — both tasks ship zero-shot seeds; incl. DeepSeek/Gemini/Cohere bake-off rows (gemma wins everything; free Gemini serving of the same checkpoint scores −0.048) |
| `20260710-v2-stage-architecture/` | 1+2 | closed | — | gemma-4-31b | local-val, not dev | **v2 architecture report**: error anatomy (labeling, not detection: T2 gold mass 0% unpredicted), oracle ceilings (0.86/0.88), zero-shot verify stages flat, decision-level judge compiles lift decision accuracy (GEPA AN 0.74→0.88) and optval macro (+0.033) but FAIL frozen-val transfer (+0.001, winner's curse) — champions unchanged |
| `20260710-233257-145267Z-sparse-tboth-both-legacy-train-723ea1b146/` | 1+2 | closed | both | TF-IDF + LinearSVC | OOF 0.5728 T1 cross-fit / 0.5832 T2; fixed-182 confirm 0.6702 / 0.6232 | **Canonical strict no-LM CPU run**: complete provenance + raw margins; T1 same-OOF threshold diagnostic is 0.6176; exact seeded rerun hashes match; see `../SPARSE_BASELINE.md` |
| `20260710-231400-297323Z-encoder-t1-camelbert-msa-both-legacy-train-smoke-fe95497542/` + `20260710-231411-214841Z-encoder-t2-camelbert-msa-both-legacy-train-smoke-897186f054/` | 1+2 | closed | both | CAMeLBERT-MSA | smoke only | **Non-generative encoder plumbing**: both real-data paths complete through backward pass, OOF decode, official scorer, provenance, and completion marker; deliberately non-comparable one-step frozen-head runs; see `../ENCODER_BASELINE.md` |
| `20260711-v3-proposal-atom-role/` | 1+2 | closed | both | gemma-4-31b | frozen-182 T2 0.6995 (adopted) / T1 champion retained | **v3 campaign report**: proposal→atom→role passes the +0.02 OOF gate on both tasks (T2 +0.0387, T1 fusion +0.0436), frozen confirmation adopts **Task 2 0.6995 vs 0.6805**; T1 fusion fails transfer (0.6745 vs 0.7024) — first pre-registered win; incl. champion run-to-run stability finding (±0.03–0.04 macro on identical paragraphs) |

The v3 campaign writes microsecond/config-hash run directories and is indexed
by its hash-bound artifacts rather than a pre-created folder. Its release
contract and exact command sequence are in `../V3_MILESTONE.md`. An LLM or encoder run
is valid only when `COMPLETED.json` is present; interrupted directories are
never inputs to aggregation or deployment.

## Registry rows and deployments → run directories

The paper's registry table (Appendix B) has 29 rows. This section names the directories
whose `metrics.json`, `REPORT.md`, or report JSON hold each row's numbers, then the
deployment twins and the later measurements. Directory names are abbreviated with `…`
where the timestamp and hash suffix are unambiguous from `ls experiments/`.

| Row | Decision | Directories |
|---|---|---|
| 1 | quote-then-align vs segment (T2) | `20260709-d3-quote-vs-segment/REPORT.md`; inputs `20260709-t{1,2}-quote-*-val/` |
| 2–4 | BFRS, MIPROv2, GEPA compiles (T1) | `20260709-d7-optimizers/REPORT.md`; `20260709-t1-{bfrs,mipro,gepa}-gemma-4-31b-paid/` (with `instructions.json`) |
| 5 | GEPA compile (T2) | `20260709-d7-optimizers/REPORT.md`; `20260710-t2-gepa-gemma-4-31b-paid-val/` |
| 6–8 | zero-shot verify stage, blind relabel stage, decision-level judge compile | `20260710-v2-stage-architecture/REPORT.md` and the `20260710-*` stage-test runs it names |
| 9 | v3 proposal-atom-role (T2) | `20260711-v3-proposal-atom-role/REPORT.md`; role folds `20260710-*-v3-role-*-outer{0..4}-inner0-*`; adoption `20260710-231102-…-v3-cv-adoption-…-task2-4e4fa7e16a/adoption_report.json`; final compile `20260710-231222-…-v3-role-gepa-…-final-inner0-bc5f4625f5/`; frozen confirmation `20260711-003552-…-v3-decomposed-…-val-87116e6559/`; dev deployment `20260711-033302-…-v3-decomposed-…-dev_in-78718d42e5/` |
| 10 | v3 Task 1 fusion | `20260710-231142-…-task1-fusion-…-b67a9b1328/report.json`; frozen confirmation in the v3 report |
| 11 | theory-variables-alone design | `20260711-081348-…-t1-signal-audit-…-n8-…/`, `20260711-081440-…-t1-signal-audit-…-n430-7d1b713d9a/` |
| 12 | tier-1 routed composition (T1) | gate `20260711-105107-…-t1-routed-k5-n430-0136977dee/`; rollouts `20260711-{100552,101957,103425,105038,105041}-…-t1-zeroshot-r{0..4}-…-train-*` (`20260711-{092229,093929}-…-r{0,1}-…-train-*` are the failed first attempts of r0 and r1, `FAILED.json`, not consumed) and `20260711-1{05157,10055,10724,11253,11820}-…-r{0..4}-…-dev-*`; encoder OOF `20260711-090448-…-encoder-t1-camelbert-msa-quarter-…-0e77e5faf4/`; encoder deployment `20260711-113659-…`; dev deployment `20260711-113730-…-t1-routed-deploy-dev_in-009d4fd89e/` |
| 13 | mix-encoder amendment | `20260711-163945-…-encoder-t1-camelbert-mix-…/`, `20260711-170107-…-msa-…/`, `20260711-172215-…-da-…/`; screens `20260711-{170117,174308}-…-t1-routed-k5-n430-*`; deployments `20260711-174851-…-t1-encoder-deploy-camelbert-mix-dev_in-…/`, `20260711-174921-…-t1-routed-deploy-dev_in-554dc13d33/` |
| 14 | routed v2 recomposition | gate `20260714-194246-…-t1-routed-v2-k5-n430-375f2468c5/`; ST seed judge `20260714-t1-judge-stage-gemma-4-31b-paid-val/` (registered as the BFRS-compiled ST judge, executed zero-shot: `judge_stage_test.py --drop-labels ST`, `compiled: null`; the folder's `config.json`/`metrics.json` describe its dev-217 pass, while the train-430 pass the gate consumes, `predictions/st_judge_train430.jsonl`, has no record of its own; OOF ST 0.5902 → 0.6429, dev 0.6923 → 0.7619); dev deployments v2 `20260714-194633-…-dd107f0d5b/`, v2.1 `20260714-202259-…-fba4d257e3/` |
| 15 | CO many-shot rank feature | `20260714-195647-…-co-judge-train-oof-…-n8-…/`, `20260714-200016-…-co-judge-train-oof-…-n430-…/` |
| 16 | T2 atom-vote self-consistency | `20260714-200743-…-t2-vote-pilot-k3-n6-…/`, `20260714-204703-…-t2-vote-pilot-k5-n86-…/` |
| 17 | discourse-forest prompts | forests `20260711-17{1237,3523}-…-discourse-forests-…/`; `20260714-{194441,195418,200827}-…-v4-structure-{flat,forest,shuffled}-…/`; `20260714-203040-…-structural-cv-comparison-…-833d98c8fb/comparison_report.json` |
| 18 | constrained relabel (T2) | `20260714-21{3215,3556,3754,3853,4125}-…-t2-crelabel-…-n86-*` |
| 19 | constrained relabel re-gate | `20260715-01{2938,3010,3019,3034,3037}-…-t2-crelabel-…-n86-*` |
| 20 | learned stacker screen (T1) | ledger entry only (public commit `3a52b50`); no run directory |
| 21 | ST contrastive verifier v2 | gate `20260714-202929-t1-st-verifier-v2-gemma-4-31b-paid-train430/`; deploy twin `20260714-210511-…-dev217/`; co-signals `20260714-{200108,200859}-…-llama-3-3-70b-…-train-*`, `20260714-201837-…-qwen3-32b-train-…/`, `20260715-00555{1,2}-…-{qwen3-32b,llama-3-3-70b-paid}-dev_in-*`; v2.2 OOF `20260715-003004-…-t1-routed-v2-k5-n430-4f30d408fa/`; dev deployment `20260715-010537-…-t1-routed-v2-deploy-dev_in-a650666233/` |
| 22 | CO contrastive specialist | `20260714-222759-t1-co-specialist-gemma-4-31b-paid-n430/` |
| 23 | AN-rescue span judge | `20260714-232555-t2-an-rescue-gemma-4-31b-paid-f0/` |
| 24 | S1 structural re-decode (T2) | gate `20260715-001221-t2-s1-structural-camelbert-quarter-n430/` (its `metrics.json` names the encoder and quote inputs); T2 encoder OOF `20260715-033553-…-encoder-t2-camelbert-msa-quarter-…-78b1f7b0d3/`; encoder deployment `20260715-042523-…-t2-encoder-deploy-…-dev_in-f04daf77d5/`; dev deployment `20260715-002802-t2-s1-deploy-dev_in/` |
| 25 | S2 learned span combiner | `20260715-003839-t2-s2-combiner-n430/` |
| 26–27 | bundle-v3 full and ensemble-only | `20260715-124435-…-t1-bundle-v3-n430-30372f1e85/`; seed ensembles `20260714-233725-…-t1-encoder-ensemble-quarter-5seed-cpu-n430-…/`, `20260714-23454{1,2,3}-…-t1-encoder-ensemble-{msa-quarter,mix,da}-5seed-fp16-n430-*`; per-seed encoder runs `20260714-21{4717,1008}-…`, `20260714-22{3217,5500}-…`, and `../kaggle/encoder-multiseed-v2/results/` |
| 28 | calibration temperature τ (T2) | `20260715-160053-t2-g1-tau-n430/` |
| 29 | transductive TAPT (T1) | `20260715-202537-…-tapt-mlm-camelbert-quarter-transductive-829-…/`, `20260716-081636-…-encoder-t1-camelbert-msa-quarter-tapt-…/`, `20260716-083652-…-t1-tapt-gate-n430-92f0689565/` |

**Test phase, pinned recipes (2026-07-27).** Quote run `20260727-043727-…-t2-quote-…-test_in-e0cb44f029/`;
direct Task 1 view `20260727-043728-…-t1-zeroshot-…-test_in-7b1ca39fa9/`; role decisions
`20260727-044517-…-v3-decomposed-…-test_in-aada058d66/`; rollouts
`20260727-{044313,044845,045955,050633,051224}-…-t1-zeroshot-r{0..4}-…-test_in-*`; co-signals
`20260727-051803-…-qwen3-32b-test_in-…/`, `20260727-052425-…-llama-3-3-70b-paid-test_in-…/`;
encoder deployments `20260727-045149-…-t1-encoder-deploy-…-test_in-c49bfe1c7b/`,
`20260727-045700-…-t2-encoder-deploy-…-test_in-d6963ed563/`; seed judge and verifier
`20260727-01{2949,3209}-…-test213/` (dev control `20260727-004607-…-dev217-control/`);
router passes `20260727-05{2907,3009,3238}-…-t1-routed-v2-deploy-test_in-*` (the last,
`b8d7aae3a3`, is the scored frozen Task 1 entry); Task 2 `20260727-005947-t2-s1-deploy-test_in/`
(dev control `20260727-003450-t2-s1-deploy-dev_in/`).

**Evaluation week (2026-07-29 and 07-30).** Four additional Task 1 encoder deployments per
target, seeded from the other out-of-fold seed runs, `20260729-22*-t1-encoder-deploy-…-{dev_in,test_in}-*`;
the probe compositions themselves have no run directory (`../EVAL_WEEK_PROBES.md`). The
`20260730-*` directories (S1 regression controls, a second-seed T2 encoder run, T2 encoder
ensembles and deployments) belong to the Task 2 uploads of 2026-07-30 that were never
pushed to the board.

**Registered measurements, not deployed.** GN guideline-native program
`20260729-1{71558,74953,82555,83456,83511,83519,92603,94331,94343,94644,94657}-…-t{1,2}-guideline-*`
(ledger section "GN"); the paper-analysis bank `20260716-133715-…-paper-analysis-bank-n430-d7e5a011ed/`.

**Records added after the campaign (2026-09-06).** The paper's placebo ablation
(§3, 0.7133) originally ran on 2026-07-15 against a temporary uniform-score
encoder folder that was not kept, leaving only the ledger sentence; it was re-run
on the real encoder run's segment layout with every score set to 1/6: input
`20260906-022503-t2-s1-placebo-uniform-scores-n430/`, gate
`20260906-022528-t2-s1-structural-placebo-uniform-n430/` (`bundle_f1` 0.7133,
delta +0.0199, 5/5 folds; the result is invariant to the constant and to the
segment boundaries because `t2_structural_gate.py` renormalizes each span's
distribution, so the run stands in for the deleted original). The corpus
constants quoted in the paper's §2 (612; 357/255; 2,975 spans; 90.73% character
coverage; per-label priors; AN base rate 0.1153; test 87/126) are recomputed from
the organizer files by `scripts/paper_corpus_counts.py` into
`paper/corpus_counts.json`.

**Camera-ready ablation (2026-09-07).** Reviewer 29cU asked for a clearer
ablation of the S1 parts than the placebo alone gives.
`scripts/t2_s1_ablation.py` runs a leave-one-out over the bundle and writes
`20260907-160206-t2-s1-ablation-loo-n430/` (paper Appendix C). It re-fits every
per-fold quantity, λ included, inside each variant, so each row scores that
system rather than reusing the full system's fit; the `full` row reproduces
0.7206 and the v3 baseline 0.6934 exactly, which is the run's control check.
Results: −calibration 0.7210 (+0.0004), −genre conditioning 0.7167 (−0.0039),
−Rule-A 0.7166 (−0.0040), −Viterbi 0.7073 (−0.0133), −blend (λ=0) 0.6507
(−0.0699). Zero API cost, no predictions written, report-only: no gate is
defined over these variants and nothing here can change a frozen system.

Two properties of the `−Viterbi` row matter when reading it (corrected
2026-09-07 after review, paper Appendix C says the same). First, the emission
blend is `(1−λ)·p_enc + λ·onehot(v3)` over a *normalized* encoder distribution,
so a per-span argmax can overturn the v3 label only when `(1−λ)(e_k − e_L) > λ`,
which is impossible at λ ≥ 0.5. That row is therefore v3 plus Rule-A and nothing
else: its 103 label changes are all Rule-A's (`n_label_changes` equals
`n_rule_a_changes` in `metrics.json`). Second, its re-fitted λ = 0.5 is not a
finding: λ ∈ [0.5, 0.9] all decode identically, and the selection loop takes the
first of that plateau because it compares with a strict `>` over an ascending
grid. The grid is `LAMBDA_GRID = 0.0 … 0.9`, so the full bundle's λ = 0.9 is the
grid's upper edge, and λ = 1 (v3 untouched) is outside it.

Read against the placebo run (`20260906-022528-…`, 0.7133), the parts do not
add. On top of v3: Rule-A alone +0.0139, the sequence and label-mass priors a
further +0.0060, real encoder posteriors the remaining +0.0073. Yet removing
Rule-A from the full bundle costs only −0.0040, because Viterbi already repairs
most of what Rule-A repairs. Leave-one-out and add-one-in shares are not
interchangeable here, and neither the genre nor the Rule-A delta clears the
σ≈0.015 composed-OOF band the gates are calibrated to.
