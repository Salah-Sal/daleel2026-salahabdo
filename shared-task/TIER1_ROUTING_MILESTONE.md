# Tier-1 Task 1 milestone: per-label routing + self-consistency + CO budget

Commissioned 2026-07-11, AMENDED with the signal-audit
findings before any tier-1 data was generated. This document pre-registers
the design, the calibration protocol, and the adoption rule.

## Evidence base (all measured, nothing speculative)

| Label | LLM champion (dev) | Encoder OOF (Kaggle) | Signal audit note |
|---|---|---|---|
| AS | 0.90 | 0.905 | saturated everywhere |
| TE | **0.79** | 0.636 | attribution is surface-detectable |
| ST | **0.67** | 0.238 | 19–28 supports can't teach an encoder |
| AN | 0.46 | **0.725** | convention; `form=instance` also recovers 0.67–0.71 unsupervised |
| OT | 0.57 | **0.769** | convention |
| CO | 0.30 | 0.31 | resisted zero-shot, supervision, AND theory variables (crossfit 0.07) — only calibration remains |

Champion run-to-run noise: ±0.03–0.04 macro (measured on identical configs).

## Amendments from the signal audit (2026-07-11)

> Correction (2026-08-06 paper audit): the commit hash `8463378` originally
> cited here does not exist in this repository. The signal-audit verdict
> commit is `e15c8f9`; this document itself first entered git in the gate
> verdict commit `c603bad`.

1. **CO gets no criterion, only a budget.** `stance=presupposed` fires on
   settled events (319 AN vs 65 CO of 857 units): annotator-CO is not
   recoverable by definition-based judging at this data scale. The design
   therefore drops any CO verifier/criterion and uses pure prior-calibrated
   ranking (below).
2. **AN routes to the encoder; no seed-prompt surgery.** The audit shows the
   AN failure is criterion-prior mismatch, but the frozen seeds stay frozen:
   the encoder leg (0.725) already embodies the convention. The label-blind
   `form=instance` detector (0.67–0.71) is a recorded FALLBACK, not a leg.
3. **Champion stays untouched.** All LLM legs use the frozen stage-0
   zero-shot program; self-consistency wraps it without modifying it.

## Pre-registered design

**Sources on the 430 train-side paragraphs (frozen 182 untouched):**

- `S_LLM`: frozen champion (`ClassifyParagraph`, CoT, gemma-4-31b-paid,
  ChatAdapter), k=5 rollouts at temperature 0.7 (`--rollout-id 0..4`),
  plus one T=0.0 reference run. Per-label vote fraction v_L ∈ {0, .2, …, 1}.
- `S_ENC`: local rerun of the verified five-fold CAMeLBERT-msa-quarter
  protocol (identical args to the Kaggle campaign; device=cpu fp32).
  Cross-fitted OOF label decisions + raw sigmoid scores per label.

**Router (fixed a priori from the evidence table — NOT fit):**

| Label | Source | Decision |
|---|---|---|
| AS, TE, ST | LLM votes | fire iff v_L ≥ θ_L, θ_L cross-fitted per label |
| AN, OT | encoder | the runner's cross-fitted OOF decision, as-is |
| CO | budget | rank paragraphs by encoder CO sigmoid score; fire top-k, k = round(0.059 · n) |

- θ_L grid {1/5, …, 5/5}, fit by fold (the encoder's fold assignment) on
  the other four folds, applied to the held-out fold — honest composed OOF.
- CO prior 0.059 = 36/612 full-train gold rate (a corpus constant, not a
  fitted parameter). Ranker = encoder CO score, pre-registered; the LLM
  vote fraction is computed as a diagnostic only.
- Composed prediction = union of fired labels; empty set legal.

**Adoption rule (pre-registered):** adopt the routed system for a dev probe
iff composed cross-fitted OOF macro-F1 ≥ (T=0 champion on the same 430)
+ 0.02. The +0.02 bar matches the campaign-long adoption convention and
exceeds the measured noise band. If adopted, ONE dev probe; no leaderboard
iteration (winner's curse rule stands).

**Deployment (only if adopted):** LLM legs = 5 rollouts on the target
input; encoder legs = train on all 430 with `final_thresholds_fit_on_all_
oof_scores`, predict the target; same router table, same k-rule with
n = target paragraph count.

## Cost ledger

6 LLM runs × 430 ≈ $6; encoder CPU rerun $0; dev probe (if adopted)
5 × 217 ≈ $3. All paid spend logged per the standing escalation policy.
