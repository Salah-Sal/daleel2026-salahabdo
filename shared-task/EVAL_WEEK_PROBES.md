# Evaluation-week probes

The paper's results section discloses a post-freeze campaign; this file records it at rule
level so that the two scored leaderboard entries can be traced. Nothing here was
pre-registered, and the paper makes no methodological claim from it.

## Protocol

When the organizers released the development references (217 paragraphs) and extended the
evaluation phase, the author waived the repository's own freeze for one evening
(2026-07-29) and ran a probe-and-revert campaign: package a variant, upload it, read the
per-label scorer output, keep or revert. Only the last upload per task counts, so the
best-measured variant was meant to be re-uploaded last. Two rules held throughout: no
overlapping or duplicated spans that would exploit the Task 2 metric, and closed-track
legality (organizer data only, open weights at or under 70B). Before any selection, the
harness reproduced both official development scores digit-for-digit from the frozen
submissions (Task 1 0.7089, Task 2 0.7247), and the released development references were
used as an offline selection set alongside the 430-paragraph out-of-fold pool.

Every probe was a recomposition of cached signals; no new LLM call was made. The inputs
were the test-phase runs of 2026-07-27 (five Task 1 rollouts, the role-decision spans, the
quote run, both encoder deployments) plus four additional Task 1 encoder deployments seeded
from the other out-of-fold seed runs (`experiments/20260729-22*-t1-encoder-deploy-*`,
`--seed-from-oof-run`), which supplied the five-seed sigmoid ensemble.

## Probes

Task 1 legs are per label; a leg not listed is unchanged from the frozen v2.2 recipe
(AS/TE/ST from vote thresholds 2/4/4 with the ST judge and verifier, AN from the encoder
or span-and-vote rule, OT from the encoder, CO from the rank-mean budget). Task 2 filters
apply to the frozen S1 output and only remove spans; they never add, merge, or move one.

| Probe | Task 1 change | Task 2 change | Dev twin T1 / T2 | Official T1 cf1 (ed / deb) | Official T2 cf1 (ed / deb) |
|---|---|---|---|---|---|
| frozen | v2.2 | S1 | 0.7089 / 0.7247 | 0.6499 (0.4595 / 0.7112) | 0.7055 (0.6195 / 0.7364) |
| 1 | AS and TE from role-decision span presence; AN from encoder or span | drop a CO/AN/OT span whose label is absent from the probe's Task 1 labels (rule B); drop a CO/OT span whose encoder segment posterior is below 0.3 (rule A) | 0.7181 / 0.7440 | 0.6574 (0.4563 / 0.7258) | 0.7154 (0.6282 / 0.7471) |
| 2 | AS back to votes; TE from span; AN: debate encoder-or-span, editorial encoder sigmoid at or above 0.5; CO: rank-mean budget of 8 per genre | rules B and A keyed to the probe-2 Task 1 labels | 0.6955 / 0.7439 | 0.6779 | 0.7291 (editorial 0.6755) |
| 3 | AN editorial sigmoid and CO ranker sigmoid replaced by the five-seed ensemble mean (threshold 0.5) | TE added to rule B | 0.7081 / 0.7448 | **0.7117** | 0.7302 |
| 4 | AN editorial threshold 0.45; CO ensemble budgets of 10 per genre | ST added to rule B (B over CO, AN, OT, TE, ST; A below 0.3) | 0.7239 / 0.7460 | 0.7113 | **0.7316** |

Per-label Task 1 readouts that drove the decisions: probe 1 AS 0.9342 (below the vote
leg's 0.9408 on test), AN 0.4878, TE 0.6992; probe 2 AN 0.5647, CO 0.1538; probe 3 CO
0.3077, AN 0.6136; probe 4 AN 0.6522, CO 0.2667 (the 10-per-genre budget added four
firings and no true positive). Task 2 span counts: 1796 in the frozen S1 output, 1635
after probe 1, 1561 after probe 2, 1564 after probes 3 and 4.

## What was scored

- **Task 1: probe 3 (0.7117).** A closing package combining the probe-4 legs with the
  probe-3 CO leg (composition arithmetic 0.7182) was prepared, but the upload meant to
  carry it held the probe-4 content by mistake, scored 0.711, and was never pushed to the
  board.
- **Task 2: probe 4 (0.7316).** Two later uploads (2026-07-30) that restored
  high-confidence encoder segments on top of probe 4 scored 0.733 and were never pushed
  either.
- The organizers' final standings sheet (2026-08-11) confirms both scored entries
  digit-for-digit: Task 1 0.7117308109 (editorial 0.5190976204, debate 0.7589418226), rank
  1 of 10 closed-track teams; Task 2 0.7315867359 (editorial 0.6838383095, debate
  0.7480736134), rank 5 of 8. `official_scores/final_standings.json` holds these values.

## What is and is not recorded

- The Task 2 filters are implemented in `scripts/t2_eval_week_filter.py`
  (`--absence-labels`, `--posterior-rule`); given the S1 output, the Task 1 labels, and the
  encoder segment scores, they reproduce a probe's Task 2 file exactly.
- The Task 1 leg swaps were composed interactively from the cached prediction files and were
  not written as a script or a run directory; the rules above are the complete
  description. `scripts/route_task1_v2_deploy.py` produces the frozen v2.2 legs but has no
  flags for the span-sourced AS/TE legs, the editorial AN threshold, or per-genre CO
  budgets.
- The prediction files these compositions read are not distributed (dataset text), so the
  probes cannot be replayed from this repository; see `../REPRODUCING.md`.
