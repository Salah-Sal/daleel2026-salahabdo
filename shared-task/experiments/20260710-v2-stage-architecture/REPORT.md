# v2 stage architecture — error anatomy, oracle ceilings, stage tests, decision-level compiles (2026-07-10)

**Question.** After every D7 prompt optimizer failed the +0.02 bar, can a
*program redesign* (verification/relabeling stages) beat the zero-shot
champions (T1 0.7024 macro-F1, T2 0.6805 span-F1)?

**Answer.** No configuration survived the pre-registered frozen-val
confirmation. The campaign's value is mechanistic: it located exactly
where the headroom is (labeling conventions, not detection), proved the
optimizers DO work at decision granularity (AN decision accuracy
0.74→0.88), and demonstrated that a 149-paragraph selection split cannot
support per-label configuration choices (optval +0.033 → frozen-val
+0.001, winner's curse). Both tasks still ship the zero-shot seeds.

## 1. Error anatomy of the champions (frozen val, stored predictions)

**Task 1** (macro 0.7024): recall is nearly saturated (ST 1.0, CO 0.9,
AS 0.95, TE 0.92); precision sinks the macro — CO 20 FP vs 9 TP
(P 0.31), ST 7 FP (P 0.53), OT 31 FP (P 0.63). AN is the lone recall
sink (26 FN, R 0.51). 77/182 paragraphs already exact; 72 more off by
exactly one label.

**Task 2** (span-F1 0.6805): **0% of gold span mass is unpredicted, for
every label** — extraction is solved; every error is a wrong label on
correctly-extracted text. Confusion matrix (gold-mass shares): gold AN →
33% called AS, 19% CO, 12% TE (only 33% AN); 75% of predicted CO mass
sits on gold text of other labels.

## 2. Oracle ceilings (official metrics, existing predictions)

| intervention (oracle) | T1 macro | T2 span-F1 |
|---|---|---|
| champion | 0.7024 | 0.6805 |
| perfect FP-judge (drop-only) | 0.8607 | — |
| perfect FN-adder | 0.7860 | — |
| perfect relabel, boundaries fixed | — | 0.8792 |

## 3. Stage tests — zero-shot same-model verification (all flat)

| stage test | Δ vs champion | anatomy |
|---|---|---|
| T2 relabel, draft-prior | −0.002 | anchored: 23/710 spans churned |
| T2 relabel, blind | −0.003 | 53/710 churned, same mistakes re-derived |
| T1 judge CO/ST/OT + AN arbiter | +0.002 | split verdict (below) |

The T1 aggregate hides a clean split: **criteria labels verify, convention
labels don't.** ST judge +0.031 / OT judge +0.035 per-label F1 (their
definitions are procedural: numbers present? discourse management?); the
CO judge killed 3 of 9 TPs and the AN arbiter churned 21 decisions net
negative (their definitions require the corpus's annotation conventions).
An LLM can verify criteria but not conventions — conventions live in the
training data, so supervision has to enter the loop.

## 4. Decision-level compiles (the conventions attack)

Design: every (paragraph, label∈{CO,ST,OT,AN}) pair on the optimizer
split becomes one binary example — `VerifyOne` student (state loads
directly into `JudgeStage`, shared `verify` attribute), `JudgeMetric`
with gold-span-quoting feedback. This restores BOTH properties whose
absence broke D7: the metric decomposes exactly per example, and
rare-label demos become mintable (~290 balanced train decisions incl.
every gold CO/AN).

Decision accuracy on 165 balanced opt-val decisions:

| judge | CO | ST | OT | AN | overall |
|---|---|---|---|---|---|
| zero-shot seed | 0.85 | 0.97 | 0.86 | 0.74–0.76 | 0.84–0.85 |
| BFRS (4 decision demos) | 0.82 | 0.97 | 0.88 | 0.80 | 0.861 |
| GEPA (DS-pro reflection, instructions-only) | 0.85 | 0.97 | 0.80 | **0.88** | 0.867 |

GEPA's +0.14 on AN decisions is the compile working exactly as designed
— corrective convention knowledge entering via reflection on span-quoted
feedback. (Caveat: balanced decision sets use random negatives; the
deployment distribution for a drop-judge is fired labels, i.e. hard
negatives.)

## 5. Deployment on optval proposals (149 paras, champion 0.6653)

| judge variant | optval macro | Δ |
|---|---|---|
| zero-shot | 0.6698 | +0.005 (replicates §3 flatness on a 2nd split) |
| GEPA | 0.6794 | +0.014 (AN +0.073 but OT judge trigger-happy: 39 drops, −0.116) |
| BFRS | 0.6912 | **+0.026** (AN +0.076, ST +0.110, OT +0.012; CO judge −0.043) |
| composed: BFRS for ST/OT/AN, CO disabled | **0.6983** | **+0.033** |

Same proposals, same architecture, same labels: the 4 decision demos are
worth +0.021 over the zero-shot judge. Composition chosen entirely on
optval; frozen val untouched by any selection.

## 6. Pre-registered frozen-val confirmation — REJECTED

Champion 0.7024 + composed configuration → **0.7034 (+0.001)**; bar was
+0.02. Per-label transfer:

| label | optval Δ | frozen-val Δ |
|---|---|---|
| ST | +0.110 | **+0.032** (transferred; 1 FP dropped, recall kept 1.0) |
| OT | +0.012 | −0.018 (reversed; 22 drops hit TPs) |
| AN | +0.076 | −0.008 (reversed; 9 drops + 17 adds wash out) |

Reading: with ~50 AN / ~60 OT instances per split, per-label deltas of
±0.03–0.08 are within cross-split sampling noise, and picking the best of
12 (variant × label) cells on a 149-paragraph set inflates the expected
gain (winner's curse). ST is the one honest survivor — +0.032 on BOTH
splits — but alone is worth ≈+0.005 macro and was not pre-registered
separately, so adopting it now would be val-fishing. **Task 1 ships the
zero-shot seed, unchanged.**

## Artifacts

- Stage tests: `20260710-t2-relabel-stage-gemma-4-31b-paid-val{,-blind}/`,
  `20260710-t1-judge-stage-gemma-4-31b-paid-val/`
- Compiles: `20260710-t1-judge-bfrs-gemma-4-31b-paid/`,
  `20260710-t1-judge-gepa-gemma-4-31b-paid-refl-deepseek-v4-pro/`
  (compiled/ gitignored: decision demos + evolved instructions embed
  dataset text)
- Selection: `20260710-t1-zeroshot-gemma-4-31b-paid-optval/` (all cache
  hits, $0), `20260710-t1-judge-stage-gemma-4-31b-paid-optval{,-judge-bfrs,-judge-gepa}/`
- Confirmation: `20260710-t1-judge-stage-gemma-4-31b-paid-val-judge-bfrs/`
- Code: `RelabelStage`/`JudgeStage`/`VerifyOne` (dspy_programs),
  `JudgeMetric` (dspy_metrics), `scripts/{relabel,judge}_stage_test.py`,
  `scripts/compile_judge.py`, `run_zero_shot.py --on optval`

## Cost

≈ $2.40 today across stage tests ($0.5), two judge compiles (~$1.0),
optval deployment runs (~$0.4), frozen-val confirmation (~$0.1),
Gemini/Cohere bake-off rows (~$0.4 of the ~$1 total logged in the D7
report). OpenRouter balance and DeepSeek balance remain healthy.

## What would actually move the score (for the record)

1. **More selection data, fewer choices.** The next configuration search
   must either pre-register ONE candidate or use cross-validation over
   the whole train side (430 paras) instead of a single 149-para optval.
2. **The ST judge is real** — fold it into any future pre-registered
   bundle rather than adopting it alone post-hoc.
3. **T2 relabeler compile** (same recipe as the judge compile, ~800 gold
   span decisions) remains unrun — the T2 oracle gap (+0.20) is twice
   T1's and the GEPA-AN result suggests conventions ARE learnable at
   decision granularity. But budget expectations against the winner's
   curse finding.
4. Orthogonal and untested: self-consistency voting (72 one-label-flip
   paragraphs), 31B+26B ensemble (57B ≤ 70B closed cap — cap semantics
   is a Jul 13 info-session question).
