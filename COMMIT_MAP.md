# Commit map

This repository's history was produced from the private development repository with
`git filter-repo`, keeping the four registration documents (`CREATIVE_HEADROOM_RESEARCH.md`,
`HEADROOM_AUDIT.md`, `TIER1_ROUTING_MILESTONE.md`, `STRUCTURAL_FOREST_EXPERIMENT.md`) and
their commits. Every other file entered in one snapshot commit from the development tree at
commit `00c0c41` (2026-09-04). Development commit hashes therefore appear in three places:
the paper's registry table, `provenance.json` (`git_commit`), and the documents themselves.

## 1. Registration-document commits (development -> public)

| Date | Development | Public | Subject |
|---|---|---|---|
| 2026-07-11 | `c603bad` | `f2a9ca7` | Tier-1 routed Task 1: composed OOF 0.6721 vs champion 0.6496 — gate passed, dev probe packaged |
| 2026-07-14 | `02e19a0` | `a673704` | Discourse-forest experiment: label-blind parser, structure modes, freeze/compare CLIs |
| 2026-07-14 | `05ac51f` | `71ea9f7` | T2 AN-rescue gate registered before run: AS/CO spans in T1-AN paragraphs judged for concrete-instance form; or |
| 2026-07-14 | `2d02ee4` | `6f283a0` | CO specialist gate REJECTED: 0.7254 < 0.7498 (CO 0.40->0.3733); third confirmation CO conventions resist in-fa |
| 2026-07-14 | `46368f9` | `25babf7` | Register T2 S1 gate (calibrated Viterbi re-decode + Rule-A) BEFORE encoder run completes; Tier C killed by clo |
| 2026-07-14 | `49228a5` | `56e8795` | T2 AN-rescue KILLED at fold-0 screen: -0.0024, judge precision 0.44 — span-level AN/AS boundary is conventiona |
| 2026-07-14 | `5867f19` | `0bce8df` | T2 creative research: label scheme identified as Webis-Editorials-16 (CO kappa 0.114 = near-chance), mass misc |
| 2026-07-14 | `5eafbde` | `0f86915` | HEADROOM_AUDIT: measured lever audit and same-day P0-P7 verdict ledger |
| 2026-07-14 | `713262a` | `2192bab` | P5 ensemble harness: mean-sigmoid seed ensembler + v2.1-rules OOF gate baseline 0.7065 |
| 2026-07-14 | `922670d` | `3a52b50` | Creative-headroom research: CO=definition confusion, ST=precision-only; stacker pilot negative (0.7084<0.7203) |
| 2026-07-14 | `d713a91` | `80d57b8` | CO forensics: 4 NOT-CO conventions induced (everyone-knows→AS, settled events→AN, cited authority→TE, motion/l |
| 2026-07-15 | `0f6b709` | `45d54bc` | Register T2 G1 gate (Menon calibration temperature tau, joint lambda-tau selection) BEFORE any fitting; gate > |
| 2026-07-15 | `4607866` | `84f1982` | T1 bundle-v3 BOTH GATES FAILED (full 0.7458/+0.016, ensemble-only 0.7436/+0.0138, vs +0.02) — closed permanent |
| 2026-07-15 | `4ae1d5b` | `b2f3271` | S1 deploy twins: T2 encoder deploy (all-430 train, dev segment scores) + structural decode deploy; Rule-A cont |
| 2026-07-15 | `53382af` | `4e049e4` | Register T2 S2 gate (confidence-selective span combiner) BEFORE any fitting; gate >= 0.7406 on the 0.7206 S1 b |
| 2026-07-15 | `56d6f30` | `91eb86f` | Register T1 bundle-v3 gate on v2.2 baseline (ensemble + TE-llama + CO-llama; fixed-sequence primary/secondary) |
| 2026-07-15 | `8385f5c` | `32d1116` | T2 S1 dev CONFIRMED 0.7247 (+0.0400 over v3 0.6847, exceeds OOF delta +0.0272) — S1 pinned as eval T2 recipe |
| 2026-07-15 | `a5c24f6` | `9c369ba` | T2 G1 REJECTED: tau curve flat (0.7179-0.7212 over tau 0-3), AN mass 0.49x at every tau — deficit is extractio |
| 2026-07-15 | `cd77061` | `8155c6e` | Register T1 TAPT gate (transductive MLM on organizer input text) BEFORE any training; OOF-measurement-only pen |
| 2026-07-15 | `d0b1e84` | `1ff964d` | T2 S2 REJECTED: +0.0013 vs +0.02 gate, flip precision 0.467 < 0.5 break-even; learned-arbitration family close |
| 2026-07-15 | `fa28fa6` | `936fec9` | T2 S1 ADOPTED: pooled OOF 0.6934 -> 0.7206 (+0.0272), 5/5 folds; CO mass 3.98x->1.78x; AN deficit remains for  |
| 2026-07-16 | `1e88530` | `dbc7e36` | T1 TAPT REJECTED: transductive MLM lifts encoder +0.0476 OOF (0.5948->0.6424) but composed +0.0066 (0.7364<0.7 |
| 2026-07-29 | `056a329` | `944819a` | REGISTERED MEASUREMENT: CFG-J1 config judge probe (5 frozen configs, n=10 stress set) |
| 2026-07-29 | `427547d` | `85fc032` | REGISTERED MEASUREMENT: GN guideline-native program (frozen before full-pool calls) |
| 2026-07-29 | `e65a65e` | `03aafd9` | CFG-J1 judge probe: run, judged, closed (C1 bf16-pin wins; thinking reachable but harmful) |
| 2026-07-29 | `e9a90e5` | `fe21ef9` | GN result: all 8 legs, verdict branch (a) at the boundary (T2 en/430 = 0.6209) |
| 2026-08-06 | `8ada947` | `a071b1b` | Audit #2: fix upload-record contradiction, coverage/AN-ratio claims, registry hashes+cells, registration-timin |

## 2. Commits cited in the paper's registry table that have no public twin

These commits changed only experiment directories, scripts, or the private submission log.
Their content is in the snapshot; the verdict lives in the run directories listed.

| Development | Date | Where the content is |
|---|---|---|
| `e15c8f9` | 2026-07-11 | `experiments/20260711-081348-706940Z-t1-signal-audit-…-n8-426a9efdf6/`, `experiments/20260711-081440-983196Z-t1-signal-audit-…-n430-7d1b713d9a/`, `scripts/audit_signal_variables.py` |
| `7127672` | 2026-07-11 | `experiments/20260711-16…-encoder-t1-camelbert-{mix,msa,da}-…/`, `experiments/20260711-17…-t1-routed-k5-n430-{0029c2eb06,dba1011304}/`, the two `…-deploy-dev_in-…` twins, `scripts/deploy_encoder_task1.py`, `scripts/train_encoder_baseline.py` |
| `1ad20a0` | 2026-07-14 | `experiments/20260714-21…-t2-crelabel-…-n86-…/` (five runs), `scripts/t2_constrained_relabel.py` |
| `848c447` | 2026-07-14 | `experiments/20260714-202929-t1-st-verifier-v2-…-train430/`, `experiments/20260715-003004-912987Z-t1-routed-v2-k5-n430-4f30d408fa/`, `scripts/st_verifier_v2.py` |
| `8e3414a` | 2026-07-14 | `experiments/20260714-19…/20…-v4-structure-{flat,forest,shuffled}-…-outer0-inner0-…/`, `experiments/20260714-203040-610148Z-structural-cv-comparison-…-833d98c8fb/` |
| `9d094de` | 2026-07-14 | `experiments/20260715-01…-t2-crelabel-…-n86-…/` (five runs) |
| `f4ef8e5` | 2026-07-14 | `experiments/20260714-195647-…-co-judge-train-oof-…-n8-…/`, `experiments/20260714-200016-…-co-judge-train-oof-…-n430-…/`, `experiments/20260714-20…-t2-vote-pilot-k{3,5}-…/`, `scripts/co_judge_rank.py`, `scripts/t2_vote_pilot.py` |
| `34324ce` | 2026-07-14 | private submission log only (dev results of routed v2 / v2.1); the scores are in the paper's results section |
| `bcd0aca` | 2026-07-12 | private submission log only (mix-encoder amendment external check); the scores are in the paper's registry table |

## 3. Development hashes cited inside the documents

| Hash | Cited in | Resolves to |
|---|---|---|
| `c603bad` | `TIER1_ROUTING_MILESTONE.md` | public `f2a9ca7` |
| `e15c8f9` | `TIER1_ROUTING_MILESTONE.md` | no public twin; see section 2 |
| `46368f9` | `CREATIVE_HEADROOM_RESEARCH.md` | public `25babf7` |
| `427547d` | `CREATIVE_HEADROOM_RESEARCH.md` | public `85fc032` |
| `0ea5b82` | `CREATIVE_HEADROOM_RESEARCH.md` | development commit that added the guideline-native program (2026-07-29); its code is in the snapshot (`src/daleel/guideline_*.py`) |
| `7924acd` | `CREATIVE_HEADROOM_RESEARCH.md` | development commit that enabled MLflow autologging and module-owned LM instances (2026-07-29); in the snapshot |
