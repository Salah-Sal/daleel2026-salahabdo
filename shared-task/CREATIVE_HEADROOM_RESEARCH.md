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

## CO forensics read (2026-07-15, all 41 unanimous FPs + 7 invisible golds + 16 seen golds)

Annotator convention, induced from gold: CO = neutral exposition of the
thing under debate — definitions ("X is/means…"), procedural or legal
frameworks described from inside the system being discussed, an entity's
stated goals, established policy decisions — plus generic commonsense
premises (informal truisms). Four systematic NOT-CO patterns behind the
41 unanimous gemma FPs:
1. "everyone-knows" rhetoric (كلنا نعلم / من المعلوم / نحن نعلم) wrapping
   a CONTESTED claim → AS (~12 cases; the marker is rhetorical assertion);
2. settled historical events/instances → AN, never CO (Benz patent ¶717,
   Russia invaded Ukraine ¶515; "settled events are AN" re-confirmed);
3. legal/scriptural citations deployed AS EVIDENCE → TE (1951 Convention
   ¶317, Quranic verse + fiqh consensus ¶920, Knesset law ¶310);
4. debate-motion restatement / scene-setting ledes → OT/AN.
Miss direction is the mirror image: gemma keys on the formal definitional
REGISTER; invisible golds are informal truisms embedded in debate flow
(¶62 humans adapt, ¶153 companies want profit, ¶553 China is in Asia,
¶879 Ukraine borders Russia — 5/7 in debates).

## REGISTERED GATE: CO contrastive many-shot specialist (frozen before any call)

- Baseline: v2.2 composed OOF **0.7298** (run `20260715-003004…4f30d408fa`,
  champion quarter encoder, CO leg = 0.40 via rank-mean(enc sigmoid,
  gemma vote fraction, span presence) top-k×2 fold-local budget).
- Specialist: gemma-4-31b-paid ChainOfThought, one binary present/absent
  decision per paragraph over ALL 430, 3 rollouts T=0.7 (rollout_id
  0/1/2), score = present-fraction. Prompt = official CO definition +
  the four NOT-CO convention rules above + contrastive exemplars, fold-
  excluded: all out-of-fold gold-CO paragraphs (~20) as CO-PRESENT and
  the first 12 (sorted pid) out-of-fold unanimous-FP paragraphs as
  CO-ABSENT. A rollout that errors after retries is excluded from that
  paragraph's fraction (neutral containment; denominator shrinks).
- Frozen composition change (the ONLY change): CO leg v3 = same fold-
  local top-k×2 budget, rank-mean over FOUR features = the three frozen
  ones + specialist fraction. Control check: the 3-feature rank must
  reproduce the baseline CO fired set exactly before the 4-feature run
  is scored.
- Gate: composed macro = 0.7298 + (CO_new − 0.40)/6 (exact, single-leg
  change) ≥ **0.7498** AND ≥3/5 fold-slice wins → adopt into v2.3 +
  deploy twin + one dev probe; else BANK for bundle-v3. One run, no
  budget/threshold/feature shopping after seeing the number.
- Context: prior zero-shot definition judge as 4th feature was REJECTED
  (pooled 0.3733 vs control 0.40); the registered bet is that convention
  rules + contrastive exemplars (examples>definitions, confirmed twice)
  flip that result. If CO adoption lands, P3 re-gates free again (missed
  by 0.0001) — triple payoff.

### GATE RESULT (2026-07-15): **REJECTED — CO 0.40 → 0.3733, composed 0.7254 < 0.7498**

Run `20260714-222759-t1-co-specialist-gemma-4-31b-paid-n430` (1,290 calls,
2 parse errors contained, control check passed). Specialist-alone
separation is real but weak: gold-CO mean fraction 0.7333 vs unanimous-FP
mean 0.5447 — gemma still rates most of the 41 FPs as CO even with the
annotators' own exemplars and the four rules in context; its sampled
reasoning violates NOT-CO rule 3 explicitly while applying it. As a 4th
rank feature the weak signal net-hurt (fold-3 slice −0.022, others
unchanged). BANKED as a negative result. **Conclusion (third independent
confirmation): CO conventions resist in-family prompting entirely —
zero-shot definitions (crossfit 0.07), definition judge (0.3733), and now
contrastive many-shot with induced rules all fail; the convention must
enter via supervision or a different model family. Family-wise gate
discipline: no further same-leg prompt-variant gates; remaining CO
evidence (cross-model vote rank feature, banked +0.026) routes through
the single bundle-v3 gate.**

## REGISTERED GATE: T2 AN-rescue span judge (frozen 2026-07-15 before any call)

Fresh v3 confusion matrix (OOF, mass-based): span-level AN recall ≈ 37%
of gold AN mass; leaks AN→AS 4.1% / AN→CO 2.4% of total mass — the two
largest coherent sinks. AN is the one criterial-form label ("concrete
instance"), per the signal audit (form=instance → 0.67–0.71 unsupervised).

- Baseline: v3 recorded pooled OOF **0.6934** (5 outer role runs).
- Candidates: recorded spans labeled AS or CO, in paragraphs where the
  v2.2 T1 OOF composition (run 4f30d408fa) fires AN. Measured: 1,314
  candidates, 197 with dominant gold AN; oracle flip = **+0.0461**.
- Judge: gemma-4-31b-paid ChainOfThought, span-in-marked-context, asks
  whether the target span is a concrete instance (event, case, personal
  experience, historical example = AN) rather than stance (AS) or shared
  premise (CO). Contrastive out-of-fold exemplars: 10 gold-AN span texts
  + 10 recorded-AS spans whose dominant gold is AS (fold-excluded).
  3 rollouts T=0.7 (rollout_id 0/1/2). FLIP to AN only on unanimous
  yes among successful rollouts (min 2 successes); errors → no flip.
- Screen (pre-registered, P3 pattern): fold-0 first; proceed to folds
  1–4 only if fold-0 relabeled F1 ≥ recorded + 0.015.
- Gate: pooled 5-fold OOF ≥ **0.7134** (0.6934 + 0.02) AND ≥3/5 fold
  wins → adopt into T2 deploy + ONE dev probe; else bank.
- One run, no rule/threshold/source-label shopping after the numbers.

### SCREEN RESULT (2026-07-15): **KILLED AT FOLD-0 — delta −0.0024 (needed +0.015)**

Run `20260714-232555-t2-an-rescue-gemma-4-31b-paid-f0` (228 candidates,
684 calls, 0 errors). 27 unanimous flips, only 12 with dominant gold AN —
judge precision 0.44 vs the ~0.75 the arithmetic requires; net slightly
negative. Folds 1–4 not run (screen rule). Refines the audit's
form-detectability finding: paragraph-level AN PRESENCE is detectable
(0.67–0.71), but the span-level AN/AS boundary is convention-laden and
does not verify in-family — consistent with D7 ("criteria verify,
conventions don't") and with today's CO result. The ledger now reads:
ST (criterial) verified twice and shipped; CO and AN (conventional)
failed every prompting attack. T2's remaining substantive play is the
supervision route: the relabeler compile over ~800 gold span decisions
(D7 evidence: GEPA lifted AN decision accuracy 0.74→0.88, but deployment
transfer failed at that data size — a compile would need its own
transfer-honest gate design).

## T2 creative research (2026-07-15; literature survey + artifact forensics)

### Framing discovery: our label scheme IS Webis-Editorials-16
Al-Khatib, Wachsmuth, Kiesel, Hagen & Stein (COLING 2016) — same six
types over clause-segmented editorials (14,313 units, 300 English
editorials; Zenodo 3254405). Consequences:
- **Human agreement per type (Fleiss κ): CO 0.114, OT 0.152, AN 0.399**,
  AS 0.613, TE 0.591, ST 0.582. CO/OT are near-chance for trained
  native annotators. The human confusion matrix mirrors our model's
  cell-for-cell (annotator CG→assumption 0.562; anecdote→assumption
  0.277). A sizeable share of our remaining 25% wrong-label mass is
  irreducible annotation noise; realistic ceiling ≈ 0.75 pooled.
  This retroactively explains every CO failure in both tasks: the
  convention is statistical, not definitional — nothing to "verify".
- Genre flow facts (Al-Khatib EMNLP 2017, 28,986 editorials): anecdote
  mass is bursty/contiguous, testimony sits between anecdote blocks —
  matches our measured gold AN self-transition 39% (base ~10%).
- The corpus itself is a translate-train augmentation source with our
  exact conventions (untried in literature; Arabic AM transfer results:
  AraBERT translate-train 0.251 vs XLM-R zero-shot 0.003) — REQUIRES
  ORGANIZER RULES CHECK for closed-track external data.

### Internal forensics (all measured, cached artifacts, $0)
- **Mass calibration is the pathology**: v3 predicts CO at 4.15× gold
  mass, quote champion at 5.0×; both starve AN at ~0.5×; all other
  labels 0.9–1.4×. Gemma-family bias, not pipeline-specific.
- Rule A (CO needs cross-system consensus, else quote's label):
  **+0.0138 pooled, deterministic, zero cost** — banked component.
- Quote-based AN rescue: +0.004 only (both systems share AN blindness).
- Two-system per-span arbitration oracle: **+0.058**.
- Genre gap OOF: editorial 0.6606 vs debate 0.7062.
- Gold transitions: AN→AN 39%, CO→CO 29%, OT→OT 26% — sequence
  structure exists; v3 decides atoms independently.

### Ranked T2 plan (composable into ONE bundled gate ≥ 0.6934 + 0.02)
All Tier-A items are zero-API and wait only on the local T2 encoder OOF
run (launched 2026-07-15, CPU fp32, same recipe as Kaggle 0.6645; emits
oof_task_2.jsonl + oof_task_2_segment_scores.jsonl):
1. **Transition+position Viterbi re-decode** over atom posteriors
   (emissions = encoder segment probs blended with LLM labels;
   6×6 transitions + start priors fit per-fold; ~40 params).
   Lit: +5.8 node F1 inference-only, +13.1 full (Widmoser EACL 2021);
   ILP +3.2 (Stab & Gurevych 2017). Attacks AN burstiness, isolated CO.
2. **Per-class posterior calibration** (logit offsets vs mass priors,
   6 params/fold; Menon ICLR 2021 logit adjustment). Attacks the
   4×/0.5× miscalibration head-on.
3. **Rule-A CO consensus** (+0.0138 already measured).
4. **Span-level linear combiner** over {v3, quote, encoder}
   distributions + position/cue features (SpanNER +0.78–1.02;
   DS@GT routed hybrid; LinkNER uncertainty routing +3–21).
5. **AN-vs-AS binary pair specialist** (CAMeLBERT head on pair-restricted
   gold, confidence-selective override per SuperICL: flips only above
   OOF-tuned τ, no-regression by construction; fine-tuning >> prompting
   for discourse conventions is the canonical IDRR result).
Tier B: QLoRA decision-classifier (Qwen3-4B class) on the ~3.5k gold
atom decisions (LoRA-Land/LlamaLens evidence; saturation 200–500 ex).
Tier C (rules check first): Webis-16 translate-train encoder pretrain.
Killed by evidence: more prompt-only relabeling (Arabic LLM span typing
0.04–0.11 vs supervised 0.25–0.35; our own 4 failures); synthetic
discourse-convention generation (published null, arXiv 2503.20588);
many-shot beyond a cheap falsification probe (fine-tuning dominates at
6-label spaces, Bertsch NAACL 2025).

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

## REGISTERED GATE — T2 structural re-decode bundle "S1"
(registered 2026-07-15 while the local T2 encoder OOF run was still
training — fold 2/5 at registration commit; NO fitting, decoding, or
scoring of any bundle component had been run.)

### Rules check outcome (Tier C killed)
Organizer rules (QatarDebate webpage, "Closed Track"): participants may
NOT use "Additional labeled datasets for training". Webis-Editorials-16
translate-train is therefore ILLEGAL for our closed-track entry (all
our models are open-weight <=70B, so we are closed-eligible; the open
track is scored separately). Tier C is parked permanently unless we
decide to file a second, open-track entry.

### Bundle definition (frozen; order fixed)
Atom lattice = the recorded v3 spans of each paragraph (5 role runs,
outer0..4, the 0.6934 baseline), sorted by start offset. The bundle is
pure relabeling: extraction mass and boundaries untouched (type swaps
preserve segmentation credit under mass-F1).

1. **Encoder span distributions.** From the local T2 encoder run
   (camelbert-msa-quarter, fold-seed 20260710, connective granularity),
   per v3 span: overlap-mass-weighted mean of segment softmax `scores`
   over segments intersecting [start,end); drop NONE and renormalize to
   the 6 labels. Spans with zero segment overlap fall back to uniform.
2. **Per-class prior calibration** (closed-form, no grid): multiply by
   (gold char-mass prior / encoder predicted char-mass prior) per label,
   tau=1, priors computed on the 4 fitting folds only; renormalize.
3. **Emission blend**: p = (1-lambda) * p_enc_cal + lambda *
   onehot(v3 recorded label). lambda per fold from grid
   {0.0,0.1,...,0.9}, chosen to maximize pooled span partial-F1 over
   the 4 fitting folds after decoding them with that fold's fitted
   transitions. Single scalar per fold; grid registered here.
4. **Viterbi re-decode**: genre-conditioned 6x6 transition matrix +
   start priors, fit on gold span label sequences (gold spans sorted by
   start) of the 4 fitting folds, Laplace alpha=1. Decode = argmax
   joint log-likelihood over the span sequence.
5. **Rule-A CO consensus** (post-decode): for each span whose decoded
   label is CO, take the quote champion's dominant label by char-overlap
   mass over the span; if a dominant label exists and differs from CO,
   relabel to it; otherwise keep CO. Quote source =
   20260710-190630-477294Z-t2-quote-gemma-4-31b-paid-alltrain-5aa0011e3b.

All fitting keyed to the v3 outer folds; the eval fold's paragraphs are
always excluded from priors, transitions, and lambda selection.

### Gate rule
Pooled OOF span partial-F1 of the bundle >= recorded + 0.02 (recorded
pooled = 0.6934, so >= 0.7134) AND >= 3/5 fold wins (relabeled >=
recorded per fold). Zero API cost, so no fold-0 screen: one full run.
If ADOPTED: build deploy twin (fit calibration/transitions/lambda on
all 430, encoder deploy inference on dev, same frozen rules) + ONE
bundled dev probe; the free P3 regate then re-runs on the new baseline.
If REJECTED: bank, no same-family retry (no lambda-grid extension, no
transition-model variants, no decode-rule swaps after seeing numbers).

### Registered caveats (honesty notes, written before results)
- Quote champion was GEPA-compiled on all-train, so its train
  predictions carry in-sample flavor; Rule-A's measured +0.0138 may be
  optimistic. Bounded: Rule-A only touches spans still CO after decode.
  The dev probe is the arbiter.
- Encoder OOF probs are out-of-fold w.r.t. the encoder's own folds
  (same seed 20260710); using them as stacking features inside v3-fold
  fitting is standard stacked-generalization practice.
- Plays 4-5 from the ranked plan (span-level linear combiner over
  {v3, quote, encoder}; AN-vs-AS pair specialist with
  confidence-selective override) are NOT in this bundle. They form a
  separate future registration ("S2") on whatever baseline stands after
  S1 — different mechanism family (learned arbitration vs decoding),
  so family-wise discipline is preserved.

### S1 RESULT — ADOPTED (2026-07-15)
Gate run 20260715-001221-t2-s1-structural-camelbert-quarter-n430:
pooled 0.6934 -> **0.7206** (+0.0272), gate 0.7134, **5/5 fold wins**.
Pre-Rule-A decode 0.7166; Rule-A added +0.0040 (38 flips). 237/3824
span labels changed. Encoder evidence contribution isolated by the
placebo ablation (uniform scores -> 0.7133): +0.0073 real. lambda=0.9
all folds; calibration weights mild (ST ~1.25-1.40, AN ~1.1). Mass
ratios: CO 3.98x -> 1.78x, ST 1.38x -> 1.07x; AN unchanged 0.49x —
the anecdote deficit is untouched and remains S2's target.
Local encoder standalone OOF 0.6562 (Kaggle GPU twin 0.6645), AN-rich
error profile (465 AN vs 36 CO segments) — decorrelation confirmed.
NEXT: deploy twin (fit on all 430, dev inference) + ONE bundled T2 dev
probe. P3 note: its 0.7133 is below the new 0.7206 baseline; the
prompt-relabel track is subsumed by S1's CO fix — any P3-on-S1 retry
would be a NEW registration at >= 0.7406, currently not planned.

### S1 deploy note — Rule-A caveat VOID (upgrade)
The registered caveat assumed the quote source was GEPA-compiled on
all-train. Its config shows it is run_zero_shot.py --program quote at
T=0 (stage0-quote-v1, no compile step) — there is no fitting and hence
no in-sample contamination in Rule-A. The dev twin
(20260711-032218-...-e955af86f9) is the identical program on dev_in.

### S1 DEV CONFIRMATION (2026-07-15)
Codabench dev: **0.7247** (P 0.6840, R 0.7704; editorial 0.6875,
debate 0.7381) vs prior 0.6847 — **+0.0400**, exceeding the OOF delta
(+0.0272) and landing +0.012 above the arithmetic prediction (~0.712).
First adoption of the campaign whose dev transfer was POSITIVE. Both
genres improved (editorial +0.0447, debate +0.0382). S1 is the pinned
eval-phase T2 recipe unless S2 clears its own gate (>= 0.7406 OOF).

## REGISTERED GATE — T2 "S2" confidence-selective span combiner
(registered 2026-07-15 after S1 dev confirmation 0.7247; NO fitting,
feature extraction, or scoring of any S2 component has been run.)

### Baseline and gate rule
Baseline = the S1 gate run's relabeled OOF spans
(20260715-001221-t2-s1-structural-camelbert-quarter-n430, pooled
0.7206, recomputed at gate time). Gate: pooled OOF span partial-F1
>= recorded + 0.02 (= 0.7406) AND >= 3/5 fold wins. Zero API cost,
single full run, no fold-0 screen.

### Frozen design
ONE multinomial logistic regression per fold (sklearn 1.9.0, lbfgs,
L2 C=1.0, class_weight='balanced', max_iter=1000), trained on the 4
fitting folds' spans that overlap gold (target = dominant gold label
by char-overlap mass; zero-overlap spans excluded from training but
eligible for override at inference). Features per span (frozen):
  1. calibrated encoder posterior (6; S1 calibration weights fit on
     the same 4 folds)
  2. v3 recorded label one-hot (6)
  3. S1 label one-hot (6)
  4. quote dominant label one-hot + none (7)
  5. T1 v2.2 composed paragraph label indicators (6; run
     20260715-003004-...-4f30d408fa)
  6. genre indicator (1)
  7. log char length (1)
  8. relative start offset (1)
  9. is_first, is_last (2)
 10. prev S1 label one-hot + none (7)
 11. next S1 label one-hot + none (7)
Continuous dims standardized with fitting-fold mean/std.
**Override rule**: replace the S1 label with the combiner argmax ONLY
when argmax != S1 label AND max posterior >= tau; tau per fold from
grid {0.50, 0.55, ..., 0.95} maximizing pooled span-F1 on the 4
fitting folds (in-sample for selection only — same convention as S1
lambda). Exact-duplicate dedup after override (S1 rule).

This single 6-way model subsumes the planned binary AN-vs-AS pair
specialist (AS->AN flips are its dominant expected action given the
0.49x AN deficit); class_weight='balanced' is the registered choice
targeting AN recall, with the tau override guarding minority
over-fire. Design point: S1's label is itself a feature, so the
combiner learns WHEN to distrust S1 — the SuperICL pattern with the
LLM+decoder stack as the base system.

If ADOPTED: deploy twin (train on all 430; tau by all-430 in-sample
selection; dev features from S1 dev output 20260715-002802-t2-s1-
deploy-dev_in, encoder deploy ...f04daf77d5, quote dev ...e955af86f9,
T1 v2.2 dev deploy ...a650666233) + ONE bundled dev probe.
If REJECTED: bank. No feature enlargement, no grid extension, no
alternative classifiers, no class-weighting variants after seeing
numbers. S2 closes the learned-arbitration family either way.

### S2 RESULT — REJECTED (2026-07-15)
Gate run 20260715-003839-t2-s2-combiner-n430: pooled 0.7206 -> 0.7218
(+0.0013 vs required +0.02), 3/5 fold wins. Flip precision 0.4672
(137 gold-overlapping flips) — just under the ~0.50 break-even for
mass-F1 type swaps. The registered risk fired: class_weight='balanced'
over-fired minorities (33 AS->CO flips re-inflated CO 1.78x -> 2.18x)
while AS->AN (47 flips) lifted AN only 0.49x -> 0.59x. Reading: the
available signals (encoder posterior, quote, T1-v2.2, position) carry
no AN-vs-AS information beyond what S1 already extracted — consistent
with Webis-16 AN kappa 0.399 and the ~0.75 ceiling at pooled 0.7206.
Banked per registration: learned-arbitration family CLOSED. T2 eval
recipe stands at S1 (dev 0.7247).
