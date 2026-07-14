# Headroom audit: levers not used, or used the wrong way

Written 2026-07-14 (12 days to dev close, 16 to final submission). This
document audits every campaign to date and identifies score levers that were
either never tried or tried in a mis-specified form. Every claim marked
**[measured]** was computed today from committed artifacts on disk (script
snippets inline); nothing measured touches the frozen 182 or dev gold.

Current standing (dev, closed, `both`): **Task 1 routed 0.6642** (leader
0.691, gap 0.0268) · **Task 2 v3 0.6847** (oracle-given-atoms
0.8792 → ~0.19 pure-labeling headroom).

Dev per-label (routed T1): AS 0.9046 · TE 0.8256 · OT 0.7344 · ST 0.6923 ·
AN 0.5882 · CO 0.2400. Because macro divides by 6, **CO alone can close the
entire leader gap** (0.24→0.45 ⇒ 0.699); AN is the second lever
(0.59→0.72 ⇒ +0.023).

---

## A. New measurements (2026-07-14, artifacts only)

### A1. The three T1 prediction sources disagree exactly where dev is weak [measured]

Sources on dev (all already on disk, no gold used): routed deployment
(`20260711-113730…/predictions/task_1.jsonl`), v3 span-derived label sets
(`20260711-033302…/predictions/task_2.jsonl`), 5 rollout files.

| Label/genre | routed fires | T2-span fires | overlap | only-T2 |
|---|---:|---:|---:|---:|
| AN/debate | 12 | 27 | 12 | **15** ← dev debate-AN recall is 0.24 |
| AN/editorial | **56** | 14 | 13 | 1 ← routed over-fires (~64% of editorials) |
| OT/editorial | **0** | 21 | 0 | **21** ← dev editorial-OT F1 is 0.00 |
| CO (all) | 13 | 81 | 7 | 74 ← T2-CO too liberal alone; usable as rank feature |

The deployed router consults exactly one source per label. The sources err
in *opposite directions per genre* for AN — textbook complementarity that a
single-source leg cannot exploit.

### A2. Multi-source consensus beats the deployed legs on OOF [measured]

Fixed (untuned) rules over the 430-pool OOF artifacts — encoder OOF
decisions, v3 outer-fold span-derived labels, r0–r4 rollout votes:

| Rule (fixed a priori) | quarter (deployed) | mix | da |
|---|---:|---:|---:|
| AN: encoder leg as-is | 0.691 | 0.740 | 0.736 |
| AN: `enc ∪ (span ∧ votes≥1)` | **0.724** | **0.764** | 0.740 |
| OT: encoder leg as-is | 0.774 | 0.825 | 0.773 |
| OT: `2-of-3(enc, span, votes≥2)` | **0.797** | 0.810 | **0.802** |

The consensus rules are *robust across all three encoders* — the property
the mix amendment lacked (single-seed argmax → −0.059 dev collapse). The
2-of-3 OT rule sits within 0.015 of every encoder's best while never being
worst; the AN union gains +0.004…+0.033 everywhere.

### A3. The CO budget was calibrated at the wrong depth [measured]

Precision–depth over the encoder CO sigmoid OOF scores (gold CO = 26/430):

| per-fold budget | quarter | mix | da |
|---|---:|---:|---:|
| k×1.0 (deployed rule) | 0.275 | 0.314 | 0.275 |
| k×1.5 | 0.242 | **0.364** | 0.242 |
| k×2.0 | **0.316** | 0.342 | 0.237 |
| k×2.5 | 0.308 | 0.286 | 0.242 |

With a weak ranker, the F1-optimal firing count exceeds the prior — the
`k = round(prior·n)` rule implicitly assumed near-perfect ranking. Widening
to k×1.5–2 is worth **+0.04–0.09 CO F1 on OOF** with zero new inference.
Head behavior differs by encoder (da p@10 = 0.50 but decays; quarter/mix
carry usable tails to depth 50): a rank-fusion of the three sigmoids + the
already-coded LLM-vote ranker (`route_task1.py:142-146`, diagnostic CO F1
0.28–0.40 by fold) + a T2-CO-span bonus is the obvious upgrade, all free.

---

## B. Considered, but used the wrong way

1. **CO budget fixed at the corpus prior** (§A3). Right mechanism
   (calibration), wrong depth, and the two stronger rankers already coded in
   `route_task1.py` were left as diagnostics.
2. **Single-source routing** (§A1/A2). The router proved per-label
   heterogeneity is the structure of the problem, then ignored that three
   orthogonal sources exist per label. The one fusion that *was* tried
   (`select_task1_fusion.py`) is a hard per-label policy switch
   (direct/spans/intersection/union) — brittle winner-take-all, and its
   frozen-val "failure" (0.6745 vs 0.7024) was measured against a champion
   recording that the same-day rerun put at **0.6613** (±0.03–0.04 provider
   noise): the confirmation itself was noise-dominated.
3. **D7 optimizers were pointed at the monolith.** GEPA/MIPROv2/BFRS on the
   full 6-label signature averaged rare-label gains away (demos drove CO
   recall 0.9→0.6; a GEPA boundary rule deleted CO 0.9→0.1). The only
   optimizer win to date (v3 role GEPA, adopted) came *after* decomposition
   to single decisions. Per-label binary compiles (a CO ranker, an AN
   verifier) were listed in D2 as option 2 and never run.
4. **The ST judge was validated twice and never shipped.** v2 report: BFRS
   ST judge **+0.032 on optval AND +0.032 on frozen-val** — the one honest
   transferable win — left on the table because the composed bundle failed.
   Dev ST is 0.6923 with precision as the known sink; this is a free
   +0.005 macro sitting in `20260710-t1-judge-bfrs-gemma-4-31b-paid/`.
5. **Self-consistency stopped at Task 1.** Champion T=0 reruns differ by
   ±0.03–0.04 macro on identical inputs — that instability is *harvestable
   variance*, and T2 is pure labeling (0% gold mass unpredicted; every error
   is a wrong label on correctly-extracted text). Rollout infra works for
   T2 (`--rollout-id`, per-atom retry ladders) but **no aggregator exists**:
   nothing votes atom roles across k replicas.
6. **Encoder legs selected by single-seed OOF argmax.** Three transfer
   failures (D7 fusion, v2 composed judge, mix amendment) share one cause:
   picking the peak of a noisy screen. The trainer has no multi-seed
   support; `sqrt`/`balanced` class weighting and `window_pool=mean` were
   implemented and never used; sigmoid scores are stored raw but never
   probability-ensembled.
7. **Deployment calibration diverges from the OOF estimate.** OOF: θ_L
   cross-fitted, CO top-k per fold. Deploy (`route_task1_deploy.py:78-111`):
   θ_L fit on all 430 *non*-cross-fitted, CO global top-k. The OOF number
   used for adoption is not an estimate of the deployed rule.
8. **Frozen-182 as the transfer arbiter.** Champion val 0.7024 → dev 0.6159
   (−0.087); composed-OOF-430 → dev tracked at −0.008 (routing) and −0.015
   (v3). The 430-OOF is empirically the honest dev predictor; the 182 should
   demote to sanity guard (its single-shot readouts carry ±0.03–0.04
   provider noise anyway).

## C. Never considered at all

1. **Cross-model vote pooling.** The bake-off selected ONE model for
   everything; nothing ever ensembled across models. Both shipped systems
   are *zero-shot seeds* — there is no compiled prompt to invalidate; any
   LADDER model votes for free. **llama-3.3-70b (closed-legal, at the cap)
   was never evaluated at all** (free-tier contention, no paid attempt);
   qwen3-32b/qwen3-14b/phi-4-14b/mistral-small-24b were killed mid-sweep;
   aya-expanse-32b untried. Newer Arabic-first checkpoints post-date the
   pool: **Jais-2-70B** (dialectal leader — debates are transcribed speech
   and debate-AN recall 0.24 is a named residual), ALLaM-34B/70B, Fanar-1-9B
   (API access was requested Jul 9 — check status).
2. **Many-shot in-context learning for rare labels.** The ≤4-demo cap is an
   optimizer-era decision; nobody ever put the **25–36 gold CO paragraphs**
   (or the ST/AN inventories) directly in context. The project's own
   evidence says conventions are learnable from examples (encoder AN 0.725;
   GEPA AN decision-accuracy 0.74→0.88) and definitions are not (theory-
   variable CO crossfit 0.07). Reframe CO as **retrieval/ranking with
   many-shot exemplars** — the budget only needs a better ordering (3/13 at
   stake), not a classifier.
3. **T2 per-label span routing.** The T1 lesson — different paradigms own
   different labels — was never applied to T2 composition (e.g. char-level
   splice: v3 spans for AS/TE/ST, alternative source for AN/OT spans; the
   Task 2 encoder that would supply those legs was never trained past an
   8-paragraph smoke).
4. **T1→T2 constrained relabeling.** The v2 relabel stage was *blind* and
   flat (−0.002). Never tried: relabel only atoms whose role contradicts
   the routed T1 label set (a +0.05-better paragraph signal), restricted to
   allowed-set ∪ NONE. Oracle for relabel-only is 0.8792.
5. **Pseudo-labeling the unlabeled organizer inputs.** 217 dev (and, after
   Jul 25, eval) paragraphs are official data; high-agreement ensemble
   pseudo-labels could ~1.5× the encoder training pool for the AN/OT legs.
   Closed-track-legal (no external data). Never discussed in any doc.
6. **Empty leaderboard cells.** All submissions to date are closed/`both`.
   Open track (proprietary LLMs legal; gemini-3-flash already measured at
   T1 0.6806 val; gpt-oss-120b/qwen3-next-80b free) and the four per-genre
   setting cells have zero entries. These are visibility wins that reuse
   existing programs nearly verbatim.
7. **Overlap/multi-role credit.** Gold contains same-offset multi-role
   spans (e.g. TE+ST "according to WHO statistics…", para 276-style);
   `ClassifyOneAtom` supports multi-role output but composition keeps one
   label per atom. Pending the organizers' answer on overlapping
   submissions (info-session question — **held Jul 13, collect answers**),
   emitting both labels on high-confidence dual atoms is small free recall.

## D. Prioritized portfolio

Ordered by expected Δdev-macro per unit risk. Every item respects the
standing protocol: pre-registered rule menu → cross-fitted 430-OOF gate →
plateau check (win on ≥3/5 folds) → ONE bundled dev probe.

### P0 — Free recomposition bundle (T1) — target dev +0.02–0.03, cost $0
*No new inference; ~1 day.*

**EXECUTED 2026-07-14 — ADOPTED.** `scripts/route_task1_v2.py` (gate run
`20260714-194246-…-375f2468c5`): composed v2 OOF **0.7102** vs deployed v1
0.6721 recomputed in-run (+0.0381 > +0.02), **5/5 fold-slice wins**; CO
0.28→0.40, AN 0.6906→0.7243, OT 0.7744→0.7966, ST 0.5902→0.6429 (zero-shot
ST judge, 5 drops all FPs, ~$0.10 API total). Deployment
`scripts/route_task1_v2_deploy.py` (θ = median of per-fold cross-fitted
values = 2/4/4, fixing the B7 mismatch) → dev zip sha256 `7a77bbd2…`,
logged pending upload.

**DEV RESULT (same day): 0.6788 macro (+0.0146 over v1).** Four changed
legs delivered externally — ST 0.7619 (+0.070), CO 0.3158 (+0.076; the
OOF k×2 curve predicted 0.316 — exact), AN 0.6309 (+0.043; debate recall
0.24→0.42) — but **OT reversed: 0.6338 vs v1's 0.7344**. Anatomy: the
2-of-3 additions fire on (span ∧ votes≥2) with the encoder silent; both
are LLM sources, and their agreement was precision-poor on dev debates
(P 0.564). The supervised encoder remains the transfer-robust OT source —
consistent with the tier-1 conventions-need-supervision finding.
**v2.1 partial revert shipped** (`--ot-source encoder`, run
`20260714-202259-…-fba4d257e3`, zip `8b2c3c5f…`): OT byte-identical to
v1's externally scored leg, all else byte-identical to v2's scored legs —
composed dev macro is exact arithmetic: **0.6955 > leader 0.691**.

| Leg | Change | OOF evidence |
|---|---|---|
| AN | `enc ∪ (span ∧ votes≥1)` | 0.691→0.724 (quarter) [A2] |
| OT | `2-of-3(enc, span, votes≥2)` | 0.774→0.797 [A2] |
| CO | k×2 budget; ranker = rank-mean(3 encoder sigmoids, LLM vote fraction) + span-CO bonus | 0.275→0.316–0.364 [A3] |
| ST | post-filter vote-fired ST through the BFRS ST judge | +0.032 ×2 validations [B4] |
| AS/TE | unchanged | — |

Also fix B7 (cross-fit the deployment calibration) inside this bundle.
Predicted composed OOF ≈ 0.693–0.70 vs deployed 0.6721; adopt iff ≥ +0.02
over the *deployed routed* baseline, else iterate legs, not the probe.

### P1 — CO retrieval program (T1) — target CO 0.24→0.35–0.45, cost ~$3–6

**EXECUTED 2026-07-14 — REJECTED as a fusion feature** (~$0.35;
`scripts/co_judge_rank.py`, run `20260714-200016-…-c469d36091`): the
many-shot judge standalone ranks CO at 0.32 top-k (beats the v1 encoder
ranker 0.28 — examples>definitions confirmed) but adds nothing over the
3-feature fusion (pooled 0.3733 vs 0.40; 4 fold ties + 1 loss) — its
signal is already carried by the vote feature (same LLM). The v2 CO leg
stays 3-feature. Dev-mode scoring skipped.
Many-shot CO ranker: for each paragraph, score CO-likelihood 0–10 given
~20 cross-fold gold-CO exemplars + the operational convention ("settled
legal/procedural background counts as CO") + hard negatives (presupposed-
stance AN cases). Use as ranker inside the P0 budget (replaces/joins the
rank-mean). Honest CV: exemplars always drawn from other folds. Gate: beats
the P0 ranker's precision@k on ≥3/5 folds. This is the single highest-
ceiling item on the board [see header math].

### P2 — T2 atom-vote self-consistency — target dev T2 +0.01–0.02, cost ~$8–12

**EXECUTED 2026-07-14 — KILLED BY ITS OWN SPEND GATE** (~$1.7;
`scripts/t2_vote_pilot.py`, run `20260714-204703-…-b931b605fe`, fold 0,
k=5 @ T=0.7): replicas [0.6595, 0.6656, 0.6603, 0.6654, 0.6598], SD
**0.0028**; majority vote 0.6668 — beats every single replica (classic
ensemble) but only +0.0024 over the recorded 0.6644 vs the pre-registered
+0.005 bar. Diagnosis: given frozen proposals, atom-level role decisions
are stable — the ±0.03–0.04 champion noise lives in extraction and
cross-day provider drift, not role labeling. The 5-fold campaign would
chase ~+0.003 T2 for ~$8: not worth it. Consequence: bundle-v3 cannot
count on a vote-upgraded span voter; it needs P5 (encoder robustification)
or P3 to reach its +0.02 gate.
k=5 role-stage replicas over frozen proposals/atoms (T=0 replicas already
decorrelate; optionally T=0.7 with the existing retry ladder guarding
whitespace floods), per-atom majority with per-label vote thresholds,
reassemble spans. Aggregator is ~50 lines next to `route_task1.py`'s vote
counter. Validate on outer fold 0 first (~$1), then all folds, +0.02 gate.
Bonus: the voted span-derived T1 labels upgrade the P0 span voter too.

### P3 — Constrained relabel pass (T2) — target +0.005–0.015, cost ~$3

**EXECUTED 2026-07-14 — REJECTED AT THE GATE BY 0.0018** (~$0.5;
`scripts/t2_constrained_relabel.py`, five runs `20260714-2132…`–`2141…`).
Free oracle first: 47 fold-0 violating spans, relabel-to-allowed ceiling
+0.040. Live: only spans whose label the v2.1 T1 set rejects are touched
(318 across folds, 0 errors); fold deltas [+0.0194, +0.0215, +0.0436,
+0.0007, +0.0092] — **5/5 positive**; concatenated OOF **0.7117 vs
0.6934 (+0.0182 < +0.02) → no dev probe** per the standing rule.
Per-label OOF: CO 0.298→0.396, ST 0.597→0.641, AN 0.480→0.500. Banked:
all artifacts cached, so the pass re-gates FREE whenever the T1
constraint source improves (P5 seed-ensembled legs → better v2.x
composition → better capture); strongest rejected candidate on file.
Second-pass relabel ONLY where atom role ∉ routed-T1 labels, choices
restricted to routed-set ∪ NONE, applied for labels where routed OOF
recall ≥ 0.8 (AS, TE, OT). Fold-0 screen before full CV. Kill fast if flat
— the blind version was, but it lacked the constraint signal entirely.

### P4 — Cross-model voters — target +0.005–0.015 T1, cost ~$4–8

**EVIDENCE RUN 2026-07-14** (~$0.9): the `:free` llama route starved again
(as in D10) — `llama-3.3-70b-paid` ModelSpec added; run
`20260714-200859-…-3b704b9a00` on the 430, 0 program errors. Standalone
macro **0.5662** (AN 0.343 — llama is weak on this task), but the marginal
VOTE value is positive on every leg it joins: AS 0.9192→0.9233, TE
0.7782→0.7917, ST 0.5714→0.6122 (cross-fitted 6-vote θ), CO rank-fusion
0.3947→0.4211 (+llama-CO as 4th feature). Naive sum ≈ +0.014 macro over
v2 — bundle-v3 material; needs P2's span-vote upgrade or a second
cross-model voter (qwen3-32b, ~$0.5) to clear the +0.02-over-v2 probe
gate. Deployment would need one llama dev run (~$0.5).

**qwen3-32b added same day** (run `20260714-201837-…-5bc5af9b3b`,
standalone 0.599): per-label voter SELECTION beats pool growth — AS is
saturated (5v 0.9192 best, 7v 0.9156), TE wants llama only (0.7917, 5/5
fold ≥), **ST wants qwen only (0.6531 pre-judge, +0.082, 4/5 folds)**,
CO wants the cross-model vote feature (0.4211, +0.026), AN/OT unmoved.
Honest bundle-v3 sum ≈ +0.015 macro over v2.1's OOF — just short of the
+0.02 gate; stacks with the P2 span-voter upgrade if that passes.
One paid zero-shot T1 run each: **llama-3.3-70b** (never evaluated),
qwen3-32b, and (if servable/licensed) Jais-2-70B or aya-expanse-32b, on the
430 + dev. Pool into the vote counts with θ refit; per-label admission only
if the added voter improves that label's cross-fitted F1 (a bad voter is
excluded per label, so downside is bounded). Also gives three fresh CO
rankers for P1's fusion.

### P5 — Encoder leg robustification (Kaggle, $0) — variance insurance
5-seed × {quarter, mix, da} mean-sigmoid ensembles for the AN/OT legs and
the CO ranker (kills the B6 single-seed curse; the trainer needs a
~10-line multi-seed loop); `sqrt` weighting for ST/CO; finally train the
**Task 2 segment encoder** properly (never run) as a future T2 routing leg.
Adopt only through P0's plateau rule.

### P6 — Fill empty cells — leaderboard presence, cost ~$5–10
Open track `both`: rerun the two seed programs with gemini-3-flash-preview
(T1 0.6806 already measured) or deepseek; per-genre closed settings: refit
θ/thresholds/budgets on genre-restricted OOF slices (debate-ST stays on
LLM votes — 6 supports can't calibrate). Mostly packaging work.

### P7 — Forest screen (T2 science) — cost ~$2, then decide
The forest artifact is frozen (4,171 atoms, 0 parser errors) but the
pre-registered fold-0 screen (flat/forest/shuffled vs baseline) never ran.
Run it; the 5-fold GEPA campaign only if the screen's causal-specificity
criteria pass. Paper value regardless of adoption.

**EXECUTED 2026-07-14 — FULL CAMPAIGN KILLED** (~$2; comparison
`20260714-203040-…-833d98c8fb`). Fold-0, optimizer=none, 0 parse failures
in all four conditions: baseline 0.6394 · flat 0.6444 · forest 0.6493 ·
**shuffled 0.6490**. The negative control did its job — forest beats
shuffled by 0.0003, i.e. the structured-prompt gain (+0.005–0.010) is
relation-vocabulary/graph-shape effect, NOT correct-attachment signal
(interpretation-matrix row "Forest = shuffled > flat"). Official gate:
`forest_beats_baseline_gate: false` (+0.0099 < +0.02), `adopt: false`.
The 20-run 5-fold GEPA campaign is not worth chasing a 0.0003 attachment
component; the clean negative-control result goes in the paper.

### Calendar fit (dev closes Jul 26 12:00 UTC)

- **Jul 14–15**: P0 composed + gated → one dev probe. P7 screen fired off.
- **Jul 15–17**: P1 CO ranker; P4 voter runs; fold into P0 rules → second
  bundled probe only if OOF says ≥ +0.02 over P0.
- **Jul 16–19**: P2 fold-0 → full CV; P3 fold-0. P5 on Kaggle in parallel.
- **Jul 20–23**: adopted T2 changes → one T2 dev probe. P6 cells.
- **Jul 23–25**: freeze; re-upload best-per-cell LAST. Eval-phase dry run.

Residual risks: leaderboard-probe discipline (one bundled probe per system,
per the standing no-probe-shopping note); rare-label fold noise (25 CO / 19
ST positives ⇒ plateau rule, not argmax); provider drift (pin OpenRouter
serving; the free-Gemini −0.048 parity failure stands).
