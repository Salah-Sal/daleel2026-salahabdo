# v3 proposal→atom→role campaign — Task 2 ADOPTED, Task 1 fusion transfer failure (2026-07-11)

The first pre-registered win of the project. The decomposed architecture
(champion `QuoteProgram` proposals → clause/connective atomization →
GEPA-compiled `SpanRoleDecision` role classifier → span reassembly) passed the
fixed +0.02 OOF adoption gate on both tasks and confirmed for **Task 2** on the
one untouched frozen-validation shot. The Task 1 fusion path passed OOF but
failed frozen transfer, so the stage-0 zero-shot classifier remains the Task 1
champion.

## Final system (what ships)

| Task | System | Frozen-val score | Previous champion |
|---|---|---:|---:|
| Task 1 | **stage-0 zero-shot classifier (unchanged)** | 0.7024 (recorded) | 0.7024 |
| Task 2 | **v3 decomposed roles (ADOPTED)** | **0.6995** | 0.6805 |

Task 2 dev/final inference follows the v3 milestone step 8 with
`--role-state .../20260710-231222-222547Z-v3-role-gepa-...-final-inner0-bc5f4625f5/compiled/role.json`;
Task 1 retains `run_zero_shot.py --task 1` and `--fusion direct` (no policy).

## Protocol (pre-registered in the v3 milestone note, unchanged during execution)

- 430-paragraph legacy-train pool; frozen 182 untouched until one confirmation.
- Five outer folds (`make_stratified_folds`, seed 20260710); GEPA candidates
  reranked on official metrics over inner-val splits drawn from outer-train
  only; each outer fold scored once.
- OOF aggregate vs same-pool stage-0 baselines; adoption bar
  `V3_ADOPTION_DELTA = 0.02` (hard constant).
- Zero-tolerance eligibility: any program/parse/unknown-token failure
  disqualifies a candidate from reranking.

## Stage 0 — frozen baselines (612 paragraphs, hash-bound)

- T1_ALL `20260710-184448-...-d4d015ac6c`: macro-F1 0.6556.
- T2_ALL `20260710-190630-...-5aa0011e3b`: span-F1 0.6633 (1 contained error:
  paragraph 745 truncates deterministically at max-tokens 6000; tolerated via
  `--max-error-rate 0.002`).
- Same-pool 430-side baselines used by the gate: T1 0.6533, T2 0.6547.

## Oracle diagnostic (GO decision)

Perfect role labels on the atomized proposals score Task 2 F1 **0.9488**
(+0.285 over the champion pipeline); coarse unatomized proposals 0.9004; Task 1
from perfect atoms 0.9924. The adopted system consumed ~14% of that headroom.

## Adapter finding (the one engineering blocker)

gemma-4-31b via OpenRouter fails DSPy `JSONAdapter` structured-output parsing
on ~2.8% of atom decisions (empty `{}` and degenerate tag-repetition loops);
the fold-0 smoke correctly refused to rerank (25/903 failures). `ChatAdapter`
emits the same `list[str]` shape with measured 1.0 compliance. The role stage
now uses ChatAdapter plus a deterministic cached retry ladder
(`rollout_id`-keyed, temperature 1.0) in `SpanRoleDecision.forward`
(commit 6640ba6). All five folds and the final compile then ran with **zero**
parse failures.

## Five-fold GEPA campaign (~42 min, ~$2.3/fold)

| Fold | Run | Eligible candidates |
|---|---|---|
| 0 | `20260710-193247-...-1447e7ced5` | 6/6 |
| 1 | `20260710-210609-...-d9995eddba` | 6/6 |
| 2 | `20260710-211213-...-1ce74f2e9a` | 5/6 |
| 3 | `20260710-215536-...-30cdb1492f` | 6/6 |
| 4 | `20260710-221215-...-49f2e2dad5` | 6/6 |

## Adoption gate — PASSED (both tasks)

Report: `20260710-231102-...-4e4fa7e16a/adoption_report.json` (complete: true).

| Readout | OOF candidate | OOF baseline | Delta | Gate |
|---|---:|---:|---:|---|
| Task 2 span-F1 (primary) | 0.6934 | 0.6547 | **+0.0387** | adopt: true |
| Task 1 span-derived macro-F1 | 0.6738 | 0.6533 | +0.0205 | adopt: true |
| Task 1 cross-fit fusion macro-F1 | 0.6969 | 0.6533 | **+0.0436** | adopt: true |

Fusion policy selected on OOF cross-fit
(`20260710-231142-...-b67a9b1328/policy.json`): CO/TE/ST/OT by intersection
with the direct classifier, AS/AN from spans.

## Final compile (adoption-authorized, 430 pool)

`20260710-231222-...-final-inner0-bc5f4625f5`: 6 candidates, all reranked on
official metrics over 1,191 held-in decisions, candidate 0 selected at 0.7327,
zero parse failures, `COMPLETED.json` bound to the adoption report.

## One frozen-validation confirmation (run `20260711-003552-...-87116e6559`)

| Readout | Frozen 182 | Champion bar | Verdict |
|---|---:|---:|---|
| **Task 2 v3 span-F1** | **0.6995** | 0.6805 | **+0.0190 — beats champion; ADOPTED** |
| Task 2 proposals (same-day rerun of champion spans) | 0.6857 | — | v3 adds +0.0137 over its own input |
| Task 1 fusion macro-F1 | 0.6745 | 0.7024 | **−0.0279 — transfer FAILURE; champion retained** |
| Task 1 direct (T1_ALL preds, same 182) | 0.6613 | 0.7024 | see stability note |

Decision-rule reading: the +0.02 bar is the OOF adoption gate (passed at
+0.0387); the confirmation is the transfer check, and Task 2 cleared the
champion on the untouched split. Task 1 fusion did not, and stage 0 ships —
recorded here as the pre-registered transfer-failure outcome, exactly like the
D7 judge compile before it.

### Champion-stability note (important for the paper)

The stage-0 Task 1 champion re-scored **0.6613** on the identical 182
paragraphs when its predictions were regenerated in the alltrain freeze
(vs 0.7024 recorded on 2026-07-09), and the Task 2 champion re-scored 0.6857
vs 0.6805. Same model, same config, same paragraphs — provider-side
nondeterminism at the API. Macro-F1 over rare labels (CO has ~10 val
positives) amplifies a couple of flipped paragraphs into ±0.03–0.04. Two
implications: (1) the fusion *did* improve on the direct predictions it
actually consumed (+0.013, directionally consistent with its +0.044 OOF win),
so the transfer failure is partly the champion's favorable 2026-07-09 draw;
(2) single-run frozen-val readouts carry ~±0.02–0.04 noise and dev-phase
scores should be expected to wobble accordingly.

## Cost

Five folds + smokes + final compile + stage-0 freeze + confirmation ≈ $13–14
of OpenRouter credit on 2026-07-10/11 (gemma-4-31b paid endpoint; deliberate,
logged escalation).

## Artifacts

All run directories are hash-chained: proposals sha256 → fold manifest sha256
→ CV contract sha256 → per-run `COMPLETED.json` → adoption report → final
compile config → confirmation config (full argv recorded). Compiled states,
GEPA logs, and predictions stay gitignored (they embed dataset text); configs,
metrics, fold manifests, provenance, and this report are committed.
