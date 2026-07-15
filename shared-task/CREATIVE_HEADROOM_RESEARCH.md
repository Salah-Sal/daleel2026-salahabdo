# Creative headroom research (2026-07-15)

Requested after P5 closed: find ideas superior to bundle-v3 (delayed) in
efficiency and performance. Method: two literature sweeps (aggregation /
shared-task winner recipes; efficiency / data-centric levers) + internal
error forensics on the champion composed OOF (0.7203, routed v2 + local
quarter 5-seed ensemble) + a zero-cost stacker pilot on cached artifacts.

## Internal forensics: where the headroom is

Composed OOF per label (gate run `20260714-233749…27adab9622`):

| Label | F1 | Gold + | Diagnosis |
|---|---|---|---|
| CO | 0.427 | 25/430 | P=0.32 R=0.64. Gemma votes 6/6 on **41 negatives**, 0/6 on **7 golds** → systematic definition confusion, not sampling noise. Encoder recall@50 only 0.48. |
| ST | 0.643 | 19/430 | **Pure precision problem**: R=0.947 (18/19 found), 19 FP among 37 fired. Votes ≥5 catch all 19 golds + 24 confident negatives. |
| AN | 0.764 | 129 | Balanced errors; union rule crude but serviceable. |
| TE | 0.778 | 111 | Balanced; +llama votes banked (0.7917, 5/5 folds). |
| OT | 0.791 | 160 | Encoder-only, stable. |
| AS | 0.919 | 325 | At label-noise ceiling. |

Macro arithmetic: +0.06 on one label = +0.01 macro. The rare labels
(CO, ST) are the only levers with ≥0.10 per-label headroom. Oracle over
all existing sources composes to ≈0.7265 — big jumps require new signal.

## Stacker pilot (screening-only, ran 2026-07-15, zero cost)

Per-label L2 logistic stackers, features = [enc ensemble sigmoid, gemma
vote fraction (6 runs), llama vote, qwen vote, span presence], outer OOF
on the frozen folds, decisions via the exact trainer threshold machinery.

- Full replacement: **0.7084 (C=1.0) vs 0.7203 hand rules → REJECTED.**
- Per-label: wins only OT 0.8062 (+0.016), AN 0.7670 (+0.003),
  ST 0.6512 (+0.008); loses CO 0.3582 (−0.07), TE 0.7521, AS 0.9159.
- Stacked CO collapses because LR trusts gemma's confident false
  positives — confirms CO is a signal problem, not a fusion problem.
- Hybrid cherry-pick of stacked OT/AN/ST would screen at ≈0.7248
  (+0.0045): inside the σ≈0.015 noise band AND argmax-on-screen — parked.

Matches external evidence pattern: learned aggregation wins shared tasks
when hand rules are naive (SMASH/NYCU-NLP, SemEval-2026 T9); ours are
already tuned per label, so the generic recipe has nothing left to take.

## Ranked ideas

### Tier A — performance, registered-gate candidates

1. **ST contrastive verifier v2** (top pick). Recall is the asset
   (18/19); attack only precision. For each of the ~37 fired paragraphs,
   a verification prompt containing the out-of-fold gold ST positives
   (~15 fit trivially in context) + the known FP pattern, conservative
   drop rule (drop only when verifier AND qwen-vote agree negative).
   Dropping 10/19 FPs at zero TP loss → ST ≈0.78 = **+0.023 macro** from
   ~40–80 LLM calls. Note current judge already drops some; v2 is
   contrastive + multi-model. Risk: TP drops — mitigate with the
   conservative rule; gate normally.
2. **CO rescue, two stages.**
   a. *Free forensics first*: read the 41 unanimous-FP + 7 zero-vote-gold
      paragraphs, identify the guideline mismatch, patch the prompt
      definition. (Text stays local; never committed.)
   b. *Contrastive many-shot specialist as a NEW ranker*: all ~20
      out-of-fold gold CO + ~10 hard negatives (the unanimous-FP kind) in
      context, binary yes/no, logprob or low-temp score; keep the
      budget-style decision layer (theory-optimal for weak rare signals —
      Lipton 2014; Lin & Lin CIKM 2023). Rare labels are the one place
      where every positive example fits in a context window. Upside: the
      7 invisible golds; +0.10 CO = +0.017 macro. ~430–900 calls.
3. **One more diverse model family as a vote column.** CLaC @
   SemEval-2026 T6: plain 3-family majority vote +6.2 macro over best
   single, biggest lift on the minority class. Externally validates the
   banked cross-model design (TE+llama, ST+qwen, CO+xmodel) — bundle-v3
   remains sound when we return to it; a 4th family is the cheapest
   extension.

### Tier B — efficiency (eval-phase cost) with neutral-to-positive quality

4. **Collapse 5 same-prompt rollouts → 1–2 calls.** Self-consistency
   gains plateau ≤+1.6% at 20 samples on modern models (arXiv
   2511.00751) and are non-monotonic (NeurIPS 2024, arXiv 2403.02419);
   temperature-K voting is a Monte-Carlo estimate of what logprobs give
   in one pass. If calls are kept, spend them on 2–3 *different prompt
   templates* at low temp (DiVeRSE evidence), not same-prompt rollouts.
   **60–80% cut of the main API bill for the eval phase.** Risk: changes
   vote semantics → full re-gate required; adopt only with slack before
   Jul 26, else keep frozen recipe.
5. **Distill already-paid LLM votes into CAMeLBERT** (FreeAL / PGKD
   pattern): gold CE on 430 + λ·KL to vote-fraction soft labels on dev
   inputs, agreement-filtered. Zero new API cost, one Kaggle cycle,
   +0.01–0.03 in literature at comparable N. Requires rules check on
   transductive dev use.
6. **TAPT**: continued MLM of CAMeLBERT on train+dev text before
   fine-tuning. HyperPartisan (N=515): +3.8 F1 but ±5.2 seed variance —
   run through the multiseed protocol. ~1 GPU-hour, zero API. Same rules
   check as #5.

### Tier C — variance reduction (compounds with everything)

7. **Rare-label threshold hygiene** (Lin & Lin, CIKM 2023): pool OOF
   scores across folds for a single per-label threshold (5× positives
   per decision), tune on smoothed-F or the micromacro surrogate,
   sanity-band against the plug-in F*/2 rule. Shrinks σ, which lowers
   the effective cost of every future gate.

### Killed by evidence

- Full learned-stacker replacement (pilot: 0.7084 < 0.7203).
- Dawid-Skene / Snorkel label models — dominated when gold exists
  (BOXWRENCH, NeurIPS 2024: MV/DS didn't beat supervised use anyway).
- Classifier chains / label powerset at 6 labels, N=430 — no positive
  shared-task evidence, known overfit modes; safe residue (other-label
  probs as stacker features) already tested in the pilot.
- Encoder test-time augmentation (≲1 pt, below gate alone).
- Buying more same-prompt rollouts of any model (plateau/non-monotonic).

## REGISTERED GATE: ST contrastive verifier v2 (frozen 2026-07-15, before any API call)

- Baseline: pinned v2.1 composed OOF **0.7065** (run `20260714-230133…986aefa1c6`,
  champion quarter encoder, --ot-source encoder). ST leg = 0.6429
  (18 TP / 19 FP / 1 FN over 37 fired), encoder-independent.
- Candidate set: exactly the 37 ST-fired paragraphs of that run.
- Eligibility co-signal (no new calls): paragraph is droppable only if
  qwen3-32b (run 201837) OR llama-3.3-70b (run 200859) zero-shot output
  lacks ST. Measured on cached artifacts: eligible = 13/19 FPs, 4/18 TPs.
- Verifier: gemma-4-31b-paid, ChainOfThought, contrastive many-shot —
  for a paragraph in fold f the prompt contains ALL gold-ST paragraphs
  from folds ≠ f (~15) and ALL fired-FP paragraphs from folds ≠ f (~15)
  as labeled exemplars, plus the official ST definition and caution.
  3 rollouts at T=0.7 (rollout_id 0/1/2), majority verdict.
  Program errors → containment: keep ST (no drop).
- Decision rule (frozen): drop ST for p iff majority-NOT-present AND
  co-signal-eligible. Nothing else in the composition changes.
- Gate: composed OOF ≥ **0.7265** (= baseline + 0.02) AND ≥3/5 fold-slice
  wins → adopt into v2.2 + deploy twin; else BANK for bundle-v3
  (same treatment as TE+llama / ST+qwen / CO+xmodel evidence).
- One run, no rule shopping, no threshold sweeps after seeing verdicts.
- Ceiling under this rule with a perfect verifier: ST 0.837 → +0.032
  macro; co-signal alone (drop all eligible) would be ST 0.718 (+0.0125,
  loses 4 TPs) — the verifier's job is confirming the 13 and saving the 4.
- Deploy note if adopted: dev/test co-signal requires one qwen + one
  llama zero-shot run on the eval input (~$2–3); exemplars = all 19 gold
  ST + all 19 train-OOF FPs.

### GATE RESULT (2026-07-15): **ADOPTED — 0.7298, 5/5 fold wins**

Verifier run `20260714-202929-t1-st-verifier-v2-gemma-4-31b-paid-train430`
(111 calls, 0 program errors, 3 of 37 paragraphs had rollout
disagreement); gate rerun `20260715-003004…4f30d408fa`. The verifier
called absent on 10 of the 13 co-signal-eligible FPs and kept all 4
eligible TPs — **10 drops, 10/10 false positives, 0 gold lost**.
ST 0.6429 → 0.7826 (+0.1397); composed OOF 0.7065 → **0.7298**
(+0.0233 ≥ +0.02), fold slices 5/5 wins. All other labels byte-identical.
First single-leg change to clear the adoption gate alone. **T1 recipe
v2.2 = v2.1 + ST verifier v2.** Deploy twin per the registration note:
exemplars = all 19 gold ST + all 19 train-OOF FPs, co-signal = one qwen
+ one llama zero-shot run on the eval input, same frozen drop rule; ONE
dev probe before pinning v2.2 for eval (revert = re-upload the v2.1 zip).

## Proposed sequencing (dev closes Jul 26)

1. Free CO forensics read (idea 2a) — informs 2b's prompt; no spend.
2. Register + run ST verifier v2 gate (idea 1) — cheapest large win.
3. Register + run CO specialist gate (idea 2b).
4. Winners + banked cross-model votes fold into the next single bundled
   dev probe (unchanged one-probe discipline).
5. Efficiency items 4–6 only with slack; #4 is an eval-phase cost
   decision, not a leaderboard play. #7 anytime, it's local-only.

Full literature notes (with all citations/numbers) are not included
here; headline sources: SemEval-2026 T9 overview
(arXiv:2604.06817), CLaC (arXiv:2605.02170), Lin & Lin CIKM 2023
(10.1145/3583780.3614996), Lipton et al. 2014 (arXiv:1402.1892),
Gururangan et al. 2020 (arXiv:2004.10964), FreeAL (arXiv:2311.15614),
PGKD (arXiv:2411.05045), self-consistency plateau (arXiv:2511.00751),
Are-More-Calls (arXiv:2403.02419), BOXWRENCH (arXiv:2501.07727),
many-shot/kNN demos (arXiv:2508.09323).
