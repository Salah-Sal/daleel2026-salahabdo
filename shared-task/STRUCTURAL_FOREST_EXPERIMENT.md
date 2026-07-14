# Controlled discourse-forest experiment

Pre-registered 2026-07-11 before generating any discourse forest or making
any structural role-model call.

## Question

Does a compact, correctly attached discourse forest improve Daleel atom-role
classification beyond (a) the adopted marked-text baseline, (b) an explicit
but relation-free local atom sequence, and (c) a graph-shape- and
relation-matched forest whose endpoints are deliberately misaligned from the
text?

The primary outcome is five-fold OOF official Task 2 F1 on the 430-paragraph
legacy-training pool. The frozen 182-paragraph confirmation split is not used
for parsing, prompt optimization, screening, or model selection.

## Frozen parser contract

Nodes are the existing `CandidateAtom` instances produced from the frozen
stage-0 quote proposals at `connective` granularity. Node IDs are assigned in
reading order (`A000`, `A001`, ...). The parser receives exact source text,
genre, and node ID/offset/text inventories. It never receives:

- gold Task 1 or Task 2 labels;
- the quote extractor's draft labels;
- label definitions or two-letter label codes;
- outer-fold membership or role-classifier predictions.

The only legal edge relations are:

- `ELABORATION`
- `CAUSE_REASON`
- `CONTRAST`
- `ATTRIBUTION`
- `SEQUENCE`
- `META_COMMENT`
- `OTHER_COHERENCE`

Outputs are canonicalized into a directed forest: valid node IDs, no
self-edges, one parent per child, no cycles, no duplicate edges, and only the
fixed relation inventory. Invalid edges are dropped and counted. A paragraph
may have several roots or no edges. A zero/one-atom paragraph is frozen
deterministically without a model call.

The parse artifact is generated once and SHA-bound to source text, quote
proposals, atomization granularity, parser model, parser signature, relation
inventory, and dependency/code provenance. Every experimental condition
consumes the exact same artifact.

## Four conditions

1. **baseline** — the adopted `ClassifyOneAtom` input exactly: marked text,
   genre, and exact target. No structural field or signature change.
2. **flat** — a separate structured signature receives the target plus up to
   two preceding and two following atom snippets. Typed relations are
   explicitly withheld.
3. **forest** — the same structured signature and flat neighborhood plus up to
   six validated target-local parent/child/sibling relation records.
4. **shuffled** — presentation-identical to `forest`; a seeded paragraph-local
   permutation relabels every edge endpoint. This preserves graph shape,
   roots, degrees, relation counts, and acyclicity while breaking attachment
   to atom text. The rendered input says `STRUCTURE_CONTROL=TYPED`, never
   `SHUFFLED`, so the role model is blind to the negative control.

The unchanged marked source-text window remains present in all conditions and
is authoritative. Structural snippets never replace it.

## Controlled variables

Within a screening or full campaign, all four conditions must share:

- model and adapter;
- proposal artifact and atomization settings;
- outer/inner fold manifests and seeds;
- optimizer and optimization budget;
- role metric and official external reranking;
- dynamic-demo count, class balancing, and context sizes;
- role CoT setting, token limit, and candidate eligibility rules.

Flat/forest/shuffled additionally share one forest SHA, forest-contract SHA,
and shuffle seed. `compare_structural_cv.py` rejects a comparison if these
invariants or exact fold memberships differ.

## Staged execution

### 1. Preflight and freeze the parser

The current preflight reports 430 paragraphs, 4,171 atoms, **385 planned
parser calls**, and 45 deterministic zero/one-atom paragraphs.

```bash
cd shared-task

uv run scripts/freeze_discourse_forests.py \
  --model gemma-4-31b-paid --track closed --setting both \
  --pool legacy-train --granularity connective \
  --proposals experiments/20260710-190630-477294Z-t2-quote-gemma-4-31b-paid-alltrain-5aa0011e3b/predictions/preds.jsonl \
  --threads 8 --max-tokens 1800 --preflight-only
```

Remove `--preflight-only` once the call ledger is accepted. Save the printed
`predictions/forests.json` as `FORESTS`. The default error-rate guard is zero:
an unresolved parser/program failure prevents a completed artifact.

### 2. One-fold no-optimizer screen

Run outer fold 0 for flat, forest, and shuffled with `optimizer=none`. Compare
them to a same-contract `optimizer=none` baseline. This screen is diagnostic;
it cannot authorize adoption.

```bash
uv run scripts/compile_span_roles.py \
  --model gemma-4-31b-paid --track closed --setting both \
  --proposals "$PROPOSALS" --pool legacy-train \
  --structure-mode forest --forests "$FORESTS" \
  --optimizer none --objective task2 --fold 0 --n-folds 5 \
  --inner-fold 0 --n-inner-folds 4 --threads 8
```

Repeat with `--structure-mode flat` and `--structure-mode shuffled`, changing
nothing else. Run or reuse a genuinely identical `baseline/optimizer=none`
fold. Then execute the comparison with one run per condition and
`--allow-partial`.

The full campaign proceeds only if the forest condition:

- has zero native role parse/program failures;
- beats both flat and shuffled on the screen's official Task 2 F1; and
- does not lose more than 0.01 to baseline.

The last allowance keeps a noisy single fold from prematurely rejecting a
relation-specific signal. It is only a spend gate, not an adoption criterion.

### 3. Five-fold GEPA campaign

If the screen passes, run folds 0–4 independently for all four conditions,
with `optimizer=gepa`, `objective=task2`, and the existing v3 light-budget
contract. The already completed v3 GEPA folds may serve as baseline only if
`compare_structural_cv.py` confirms every common field and fold manifest is
identical.

```bash
uv run scripts/compile_span_roles.py \
  --model gemma-4-31b-paid --track closed --setting both \
  --proposals "$PROPOSALS" --pool legacy-train \
  --structure-mode forest --forests "$FORESTS" \
  --optimizer gepa --objective task2 --fold 0 --n-folds 5 \
  --inner-fold 0 --n-inner-folds 4 --auto light --threads 8
```

Repeat for every fold and control. Do not change the forest, shuffle seed,
prompt budget, demos, contexts, or proposal artifact between runs.

### 4. Aggregate without averaging fold F1s

```bash
uv run scripts/compare_structural_cv.py \
  --baseline-runs /baseline/fold0 /baseline/fold1 /baseline/fold2 /baseline/fold3 /baseline/fold4 \
  --flat-runs /flat/fold0 /flat/fold1 /flat/fold2 /flat/fold3 /flat/fold4 \
  --forest-runs /forest/fold0 /forest/fold1 /forest/fold2 /forest/fold3 /forest/fold4 \
  --shuffled-runs /shuffled/fold0 /shuffled/fold1 /shuffled/fold2 /shuffled/fold3 /shuffled/fold4
```

The aggregator concatenates source-bound OOF predictions and calls the exact
official scorers. It also reports per-label, genre, paragraph-length, decision
accuracy, and confusion results.

## Adoption rule

Adopt the forest architecture only if the complete OOF report satisfies all
of the following:

1. `forest_task2_f1 >= baseline_task2_f1 + 0.02`;
2. `forest_task2_f1 > flat_task2_f1`;
3. `forest_task2_f1 > shuffled_task2_f1`;
4. all winning outputs have zero program/role parse failures.

Condition 1 is the standing project architecture gate. Conditions 2 and 3
are causal-specificity checks: they prevent attributing a gain to explicit
atom order, extra prompt text, relation vocabulary, or graph shape alone.

Task 1 is secondary and cannot override a Task 2 rejection. No final compile
or deployment bundle is generated for a structural condition until this gate
passes; the compiler explicitly rejects `--final` for these ablations.

## Interpretation matrix

| Result | Interpretation |
|---|---|
| Forest > flat and shuffled | Correct relation attachment carries useful signal |
| Forest = shuffled > flat | Relation words/graph shape help; attachment claim unsupported |
| Forest = flat > baseline | Explicit atom neighborhood helps; tree unnecessary |
| Flat > forest | Parser relations distract from already sufficient marked text |
| All structured < baseline | Extra structure is prompt noise or optimization burden |

No result is read against the already consumed frozen 182 split during
selection. A new development/evaluation probe occurs only after full OOF
adoption and under the project's existing submission policy.
