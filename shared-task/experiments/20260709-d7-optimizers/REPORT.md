# D7 stages 1–3 — prompt-optimizer bake-off on Task 1 + Task 2 (2026-07-09)

**Question:** which optimizer earns the compile budget for the remaining
cells — BFRS (demos only), MIPROv2 light (joint instructions+demos), or
GEPA light (reflective instruction evolution with our span-grounded
feedback metric)? Stage 0 zero-shot baselines (frozen 182-para val,
gemma-4-31b): **T1 0.7022 / T2 0.6805**.

## Protocol

- **Optimizer-internal split** (`daleel.splits.optimizer_split`, seed
  20260710): the 430 train-side paragraphs → 281 opt-train (bootstrap /
  reflection source) + 149 opt-val (candidate selection; 6 ST, 9 CO,
  62 editorial / 87 debate). **The frozen 182-para val is never seen by
  any optimizer** — it remains the paired selection set for every
  compiled-vs-zero-shot comparison, scored with the official
  `daleel.metrics` ports.
- **Call-site parameters** (D8): one graded metric per task everywhere;
  bootstrap demo gate `metric_threshold=1.0` (T1) / `0.9` (T2 — gold
  spans keep whitespace edges that trimmed aligned quotes can't
  reproduce, so 1.0 is unreachable); ≤4 bootstrapped demos; labeled
  demos 4 for T1, 0 for T2 (quote-program gold examples carry no `adus`
  field to render).
- **Task model:** gemma-4-31b (paid route, temp 0, max_tokens 6000).
  Closed-track-pure arms self-teach (gemma bootstraps/reflects gemma).
  DeepSeek-v4-pro (key added 07-09) is available as
  teacher/reflection — **open-track / pending the Jul 13 ruling on
  compile-time-only proprietary models**; run as a comparison arm, never
  as the closed headline.
- **Decision rule:** paired comparison on the frozen val, official
  scores; actionable bar 0.02 (D8 noise floor: unpaired SD ≈ 0.025).

## Optimizer budgets (verified in the 3.3.0b1 clone)

| Optimizer | Search space | Call budget (this config) |
|---|---|---|
| BFRS | demo sets (random search over ≤4 bootstrapped + ≤4 labeled) | (8+3) candidates × 149 opt-val evals ≈ 1,640 + bootstrap |
| MIPROv2 light | 6 demo sets × (3 instructions + demos) via TPE over 10 trials | valset capped at 100; ~35×10 minibatch + ~3×100 full + ~15 proposal calls ≈ 800 |
| GEPA light | instruction text only (Pareto over per-instance val scores) | 4×149 + 380 ≈ 976 metric calls + ~10 reflection calls (1-predictor formula) |

Notable 3.3.0b1 mechanics that shaped the setup: BFRS defaults `valset=trainset` if unset (we always pass opt_val
explicitly); `metric_threshold=0.0` would be treated as *unset*
(truthiness check in `bootstrap.py`); Evaluate reports percentages of
the per-example proxy; MIPROv2's old `requires_permission_to_run`
confirmation is gone (headless-safe) but it needs `optuna` (added to
pyproject); GEPA requires the external `gepa` package (present), a
5-arg metric (ours already is, by D8 design), and exactly one of
`auto`/`max_metric_calls`; GEPA never touches demos — instructions only;
reasoning-model reflection LMs returning `{text, reasoning_content}`
dicts are unwrapped by `stripped_lm_call` (so deepseek-v4-pro plugs in
directly).

## Results — Task 1 (frozen 182-para val, official macro-F1, paired)

| run | optimizer | instructions | demos | official macro-F1 | ST/CO recall | note |
|---|---|---|---|---|---|---|
| baseline | none (seed) | seed (b) | 0 | **0.7024** | 1.0 / 0.9 | stage 0 (proxy_mean 0.7355) |
| `t1-bfrs` | BFRS | seed (b) | 4 boot | 0.6601 | 0.875 / 0.7 | **REJECTED** −0.042; proxy_mean 0.6969 |
| `t1-mipro` | MIPROv2 light | **seed kept** | 3 boot + 1 labeled | 0.6616 | 0.875 / 0.6 | **REJECTED** −0.041; proxy_mean 0.7052 |
| `t1-gepa` | GEPA light (gemma self-reflection) | **seed returned** | 0 | 0.7024 (= baseline by identity) | 1.0 / 0.9 | **NO-OP**: 6 rewrites, none beat seed |
| `t1-gepa-refl-ds` | GEPA light (deepseek-v4-pro reflection) | **evolved** | 0 | 0.6828 | 0.875 / **0.1** | **REJECTED** −0.02, but see anatomy below |
| `t1-gepa-corepair` | evolved + manual CO repair | evolved, 2 clauses excised | 0 | 0.7055 | 0.875 / 0.5 | stage 3c: **tie** (+0.003 < 0.02 bar); ST 0.824 / AN 0.667 kept, CO only half-recovered (F1 0.303) |

**GEPA-3b anatomy (deepseek-v4-pro reflection).** The only optimizer
that produced an internal instruction win (candidate 5: 0.7596 opt_val
aggregate vs seed 0.7367). The rewrite adds per-label boundary rules and
**embeds three worked examples — two explicitly corrective** (cases the
model had gotten wrong, with the correction explained): a capability
bootstrap demos structurally lack, since the perfect-score gate can only
mint success cases. Frozen val: micro-F1 0.7941 (best of ANY run),
ST F1 0.696→**0.875**, AN +0.08 — and **CO recall 0.9→0.1** (F1 0.125).
Cause is legible in the text: “If the statement … is being used to
support an argument, label it AS, not CO” — a rule that logically
deletes CO, since citing shared premises to support arguments is what
common ground is for. Per-example proxy couldn't see it (dropping CO
from a 3-label set costs ⅓ of one example; the corpus macro charges ⅙
of the whole score). Stage 3c below excises the two CO-demoting clauses
(surgical, disclosed) and re-evaluates. NOTE: the evolved instruction
embeds verbatim training paragraphs → instruction dumps now live in
gitignored `compiled/`, never in git.

**BFRS post-mortem.** Internal picture was honest — best candidate
74.46 proxy vs 73.67 zero-shot (+0.8, within noise across the 11
candidates: range 69.9–74.5) — but the frozen val says −0.042 official,
replicating MIPRO's demo damage almost exactly (0.6601 vs 0.6616) with
an independently-searched demo set. Demo composition: 4 bootstrapped,
again **zero ST/CO** — with 13 ST paragraphs in opt_train and a
perfect-score mint gate, rare-label demos are structurally near-
impossible. Convergent evidence across both demo optimizers: few-shot
demos suppress rare-label recall on gemma-4-31b (CO recall 0.9→0.7
in both) and the internal proxy cannot see it.

**GEPA post-mortem (gemma self-reflection).** 924/976 rollouts, 6
candidate instruction rewrites over 6 iterations; the seed stayed on top
of the internal aggregate throughout (0.7367 proxy on opt_val vs 0.6785
for the final candidate). GEPA returned the seed program unchanged —
the D4 audit-informed seed is already past what gemma-reflected rewrites
reach. Candidate texts are recoverable from `gepa_logs/` (gitignored)
if the paper wants the diff. Stage 3b asks whether a stronger reflector
(deepseek-v4-pro) can do better with identical budget and seed.

**MIPROv2 post-mortem.** Internally it looked like a win: best full eval
77.84 vs 74.8 for the seed program (proxy %, on a 100-para subsample of
opt_val). None of the 3 proposed instruction rewrites beat the seed text
(the winner keeps seed instructions), so the entire internal gain came
from 4 demos — 3 bootstrapped + 1 labeled, all AS-heavy, **zero ST/CO
demos**: the `metric_threshold=1.0` gate admits only paragraphs the
model already answers perfectly, which skews demos easy. On the frozen
val the gain inverted on both scales (proxy 0.7052 < 0.7355; official
0.6616 < 0.7024): demos made the model conservative exactly on the rare
labels that decide macro-F1 (CO F1 0.462→0.333, ST 0.696→0.636; CO
recall 0.9→0.6). Two lessons: (1) internal optimizer scores do not
transfer — the frozen-val paired eval is the only arbiter; (2) for this
already-strong model, few-shot demos are a rare-label liability, not an
asset.

## Results — Task 2 (winning recipe, QuoteProgram)

| run | optimizer | official span-F1 | align telemetry | note |
|---|---|---|---|---|
| baseline | none (seed) | 0.6805 | 694 exact / 0 unaligned | stage 0 |
| gepa-refl-ds-pro | GEPA light, DS-pro reflection | 0.6835 (+0.003) | 659 exact / 0 unaligned | **REJECTED** (+0.02 bar) |

The one optimizer/task pairing where internal selection should have
transferred (pair-sum proxy ρ=0.990) still failed: 96 min, 11 iterations,
several accepted Pareto merges, instructions 2k→8.7k chars — and +0.003
official. Per-label the evolved prompt *reshuffles* rather than lifts:
ST +0.03, CO −0.06 (the DeepSeek-reflection CO-conservatism signature
again), AN/OT −0.03/−0.04. The evolved 8.7k-char prompt also costs 2.6×
the inference wall-clock (657 s vs 247 s per val pass). Compile crashed
AFTER saving state ("cannot schedule new futures after shutdown", a
teardown race — artifacts intact). **Task 2 ships the zero-shot seed.**

## Side experiment — DeepSeek v4 as task model (requested 07-09)

Zero-shot frozen-val rows for the two DeepSeek API models (proprietary →
open track only; primarily calibrates their value as teachers):

| model | T1 macro-F1 | T1 ST/CO rec | T2 span-F1 (quote) | T2 errors | quote fidelity |
|---|---|---|---|---|---|
| gemma-4-31b (closed) | **0.7024** | 1.0 / 0.9 | **0.6805** | 0 | 694 exact / 0 unaligned |
| deepseek-v4-pro | 0.681 | 1.0 / 0.7 | 0.6371 (~0.65 corrected) | 9/182 dropped | 664 exact / 0 unaligned |
| gemma-4-26b (closed) | 0.6492 | 0.875 / 0.9 | 0.6762 | 0 | 568 exact / 21 fuzzy |
| deepseek-v4-flash | 0.644 | 0.875 / 0.6 | 0.6534 (correction unreliable) | 19/182 dropped | 536 exact / 0 unaligned |

- **The teacher does not beat the student.** gemma-4-31b wins both tasks
  outright. DS-pro wins micro-F1 (0.780 vs 0.764) and several frequent
  labels (TE +0.036, ST +0.032) but is *more* conservative on CO
  (recall 0.7 vs 0.9) and weaker on AN — macro pays. On T2 both DeepSeek
  models under-extract (~550–670 quotes vs gemma's 710): recall 0.63–0.65
  vs 0.73.
- **Quoting fidelity is not the reasoning model's problem** — near-zero
  fuzzy/unaligned. Extraction coverage is.
- **DeepSeek endpoint quirk:** DSPy's ChatAdapter→JSONAdapter fallback
  sends `response_format`, which the DeepSeek API currently rejects
  ("This response_format type is unavailable now") — 5–10% of T2
  paragraphs (whose first ChatAdapter parse failed) died to it, scored
  as empty predictions. Any future DeepSeek *task*-model use needs an
  adapter pin. Irrelevant for teacher/reflection roles (free-text).
- Consequence for the open track: an LM swap to DeepSeek is NOT a free
  upgrade; the open-track edge must come from elsewhere (ensembling,
  external data). Consequence for teaching: DS-pro's rare-label
  conservatism means its reflection feedback won't inject the ST/CO
  aggressiveness gemma lacks — temper stage-3b expectations.

## Side experiment — Gemini + Cohere API bake-off rows (requested 07-10)

New keys (GEMINI_API_KEY, COHERE_API_KEY) → cheapest-first bake-off rows,
all zero-shot frozen-val, all open-track-only models unless noted:

| model | T1 macro-F1 | T2 span-F1 (quote) | note |
|---|---|---|---|
| gemma-4-31b (closed, OpenRouter) | **0.7024** | **0.6805** | champion, unchanged |
| gemini-3-flash-preview | 0.6806 | 0.6388 | best non-gemma T1; T2 quote fidelity is the worst measured (326 fuzzy vs 243 exact — it paraphrases while quoting); thinking pinned via `reasoning_effort=minimal` (ModelSpec.lm_kwargs) else thinking bills as output at $3/M |
| gemini-3.1-flash-lite | 0.6536 | 0.6125 | fast (45–58 s/val) and format-clean; over-fires CO (P≈0.13) |
| command-a (111B, open weights but over 70B cap) | 0.6317 | 0.6087 | faithful quoter, weak argumentative labeling (T2 ST 0.43, AN 0.29) |
| command-a-plus-05-2026 | 0.6102 | (cancelled) | 2026 flagship regresses vs command-a; 12× slower (739 s) |
| gemma-4-31b via **free** Gemini serving | 0.6545 | (cancelled) | **parity FAILURE: same checkpoint −0.048 vs OpenRouter** and 51 min vs ~4 (free-tier limits, 2 transport errors). $0 compiles via Gemini are dead: quantized/differently-batched serving shifts greedy decoding on exactly the borderline calls macro-F1 lives on |
| gemini-2.0-flash | (cancelled) | (cancelled) | older tier heavily rate-limited; abandoned after ~40 min |

Catalog notes: gemini-2.5-flash-lite 404'd on 07-09, listed again 07-10 —
lite-tier availability is flaky. Cohere serves `c4ai-aya-expanse-32b`
(open, 32B, Arabic-tuned — the one closed-track-eligible Cohere model,
untried) and `command-r7b-arabic-02-2025`. Gemini 3+ deprecates
`temperature` as a sampling knob (litellm warning) — our
determinism-by-temp-0 assumption has a shelf life on that provider.

## Findings (Task 1 — final)

1. **The zero-shot audit-seeded program stands.** Every optimizer
   variant failed the pre-registered paired bar (+0.02 official on the
   frozen val): BFRS −0.042, MIPROv2-light −0.041, GEPA-self ±0 (seed
   returned), GEPA-DeepSeek −0.02, GEPA-DeepSeek+manual-CO-repair
   +0.003 (tie). The Task 1 submission program remains the stage-0 seed.
2. **Demos are a rare-label liability on this model/task.** Two
   independently-searched demo sets caused the same damage (CO recall
   0.9→0.7/0.6). Mechanism: the `metric_threshold=1.0` mint gate
   selects only paragraphs the model already answers perfectly →
   demos over-represent easy frequent-label cases → the model
   recalibrates toward "typical" answers. The internal per-example
   proxy is structurally blind to this (D8's no-decomposition warning,
   observed live).
3. **Instruction evolution needs a reflector stronger than the
   student.** Gemma-reflected GEPA never beat the seed; DeepSeek-
   reflected GEPA produced a genuinely better instruction on 5/6 labels
   (ST +0.18, micro-F1 best of all runs) that deleted CO with one
   over-general boundary rule. Manual surgery recovered half of CO;
   stopped there to avoid val-fishing (one surgery iteration was
   pre-committed, further hand-tuning on the frozen val would
   overfit the selection set).
4. **Optimizer-internal scores never transferred on Task 1.** MIPRO
   internal +3.0 → official −4.1; BFRS internal +0.8 → official −4.2;
   GEPA-DS internal +2.3 → official −2.0. Task 2's near-decomposable
   metric (ρ=0.990) is the reason to still try GEPA there.
5. **Paper assets minted:** the demo-gate selection-bias argument, the
   corrective-examples-vs-success-demos contrast, the CO-deletion
   anatomy with legible one-clause attribution, and per-label
   complementarity across GEPA candidates (Pareto union 0.84 vs 0.74
   single-best — D11 ensembling fuel).

## Caveats

- Optimizer candidate selection happens on the per-example proxy
  (percentage of set-F1 / pair-sum F1), not the official corpus metric —
  D8 measured ~14% pair-flips between the two, so the compiled program
  chosen by an optimizer is not guaranteed to be the best on the
  official score; the frozen-val paired eval is the real arbiter.
- Smoke artifacts: `20260709-t1-bfrs-gemma-3-4b/` (tiny-subset BFRS on
  the 4B, plumbing validation only) and
  `20260709-t1-t1-bfrs-gemma-3-4b-val8/` (its --compiled eval smoke).

## Cost

OpenRouter (gemma task-model calls): session start $3.92 → $5.48 after
all five Task 1 compiles + five frozen-val evals (≈ **$1.56**; smoke
≈$0.02, BFRS ≈$0.55, MIPRO ≈$0.35, GEPA×2 ≈$0.45, evals ≈$0.20).
DeepSeek (4 zero-shot val runs + GEPA reflection): ≈ **$1.40**
($18.62 balance remains). T2 GEPA compile + its eval: ≈ **$0.65**
(OpenRouter at $6.04 mid-compile). Gemini/Cohere bake-off rows: ≈ $1
across the two new keys.

## Next steps

Superseded by the 2026-07-10 v2 architecture campaign — see
`experiments/20260710-v2-stage-architecture/REPORT.md` (error anatomy,
oracle ceilings, stage tests, decision-level judge compiles, and the
optval→frozen-val transfer failure). Both tasks still ship the zero-shot
audit-seeded programs (T1 0.7024, T2 0.6805).
