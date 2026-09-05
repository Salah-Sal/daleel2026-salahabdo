# D10 Base-Model Bake-off — Zero-Shot, Both Tasks (2026-07-09)

**Decision: `gemma-4-31b` is the base model for both tasks** (user
decision 2026-07-09: "stick with the gemma model"; fully consistent with
the pre-registered D10 criterion below). Runner-up **`gemma-4-26b`** gets
the D10 ordering-stability check (one BFRS compile) before stage-2/3
budget is committed. All optimizer stages (D7) now target gemma-4-31b.

Inputs: the `20260709-*-val/` experiment folders (one per run; committed
`config.json` + `metrics.json`, gitignored predictions). Regenerate the
table anytime with `uv run scripts/bakeoff_table.py`.

## Protocol (pre-registered as design item D10)

- **Identical seed program for every candidate, zero per-model tuning.**
  Task 1: multi-label ChainOfThought (`daleel.dspy_programs`, audit-informed
  seed instructions). Task 2: segment-then-classify, connective granularity
  (segmenter oracle ceiling 0.920). Temperature 0, `max_tokens` 4000,
  default ChatAdapter (JSONAdapter auto-fallback).
- **Eval set:** the frozen 182-paragraph local val split (seed 20260709,
  stratified genre × ST × CO, D9-cleaned gold; 8 ST / 10 CO / 21
  empty-label paragraphs). All comparisons are paired on this exact set.
- **Gates and scores:** G1 = format compliance ≥ 0.98 (Task 2 counts
  segment-count mismatches); G2 = manual Arabic-sanity read of reasoning
  traces; S1 = official metric (parity-tested ports in `daleel.metrics`);
  S2 = zero-shot ST/CO recall (rare-label headroom).
- **Provider:** OpenRouter only (sole funded key). `:free` routes where
  responsive; paid routes otherwise (approved escalation — the whole
  bake-off cost ≈ $3.8 including the multi-family sweep).

## Results (official scores on the frozen val; sorted by Task 1)

| model | B | track | T1 macro-F1 | T1 G1 | ST rec | CO rec | T2 span-F1 | T2 G1 |
|---|---|---|---|---|---|---|---|---|
| **gemma-4-31b** | 31 | closed | **0.702** | 1.00 | 1.00 | 0.90 | 0.654 | 0.98 |
| qwen3.6-27b ⚠ | 27 | closed | 0.653 | 1.00 | 1.00 | 0.70 | 0.575 | 0.89 |
| **gemma-4-26b** | 26 | closed | 0.649 | 1.00 | 0.88 | 0.90 | **0.663** | 0.96 |
| gemma-3-12b | 12 | closed | 0.631 | 1.00 | 0.88 | 0.90 | 0.602 | 0.85 |
| ministral-14b ⚠ | 14 | closed | 0.612 | 1.00 | 0.75 | 0.80 | 0.563 | 0.89 |
| gemma-3-27b | 27 | closed | 0.606 | 1.00 | 0.88 | 0.80 | 0.616 | 0.91 |
| qwen3-8b | 8 | closed | 0.594 | 1.00 | 1.00 | 0.80 | 0.573 | 0.91 |
| qwen3-30b-a3b | 30 | closed | 0.593 | 1.00 | 1.00 | 0.90 | 0.610 | 0.95 |
| nemotron-nano-30b | 30 | closed | 0.592 | 0.94 ✗G1 | 1.00 | 0.50 | — | — |
| gpt-oss-20b | 21 | closed | 0.582 | 0.99 | 0.88 | 0.40 | 0.640 | 0.98 |
| nemotron-nano-9b | 9 | closed | 0.543 | 1.00 | 1.00 | 0.60 | 0.542 | 0.81 |
| gemma-3-4b | 4 | closed | 0.533 | 1.00 | 0.88 | 0.50 | 0.530 | 0.59 |
| llama-3.1-8b | 8 | closed | 0.474 | 1.00 | 0.62 | 0.80 | 0.474 | 0.60 |
| command-r7b | 8 | closed | 0.436 | 1.00 | 0.62 | 0.40 | 0.508 | 0.66 |

⚠ = weights-license **to be verified** before any closed-track use
(release postdates verifiable knowledge). Gemma-4 rows ran on the paid
route of the same open weights (`-paid` suffix in the raw folders); the
`:free` routes were upstream-contended all day.

**Why gemma-4-31b:** decisive Task 1 lead (+0.049 over the next model —
the pre-registered actionable bar is 0.02 paired); Task 2 statistically
tied with gemma-4-26b (0.654 vs 0.663, gap 0.009 < noise); best combined
rare-label profile (ST recall 1.00, CO recall 0.90); G1 clean on both
tasks; G2 spot-read shows high-quality Arabic-grounded reasoning (verbatim
Arabic quotes, genre-aware OT handling, correct ST boundary reasoning).
One base model for both tasks keeps every compile/artifact/paper cell
simpler; nothing in S3-style evidence dissents.

Context anchors (not directly comparable — different eval sets): dev
leaderboard baseline 0.358, leader 0.691. Zero-shot local-val 0.702 with
no demos, no optimizer, no self-consistency suggests substantial headroom.

## Runs not completed

- **Killed on the lock-Gemma decision** (chains stopped mid-sweep):
  qwen3.5-9b, qwen3-14b, phi-4-14b, mistral-small-24b, qwen3-32b,
  qwen3.5-35b-a3b, nemotron-nano-30b Task 2.
- **granite-4.1-8b:** single upstream (WandB) returned 429 even on the
  paid route for >1 h — dropped.
- **llama-3.3-70b, llama-3.2-3b:** `:free` upstreams contended all day;
  no paid attempt before the decision closed the question.
- **Open-track >70B** (gpt-oss-120b, qwen3-next-80b, hy3,
  nemotron-3-super-120b — all with `:free` routes): deferred; run against
  fresh free quota if the open track is contested.

## Findings

1. **The Gemma family dominates this task at every size**, consistent
   with its multilingual pretraining mix: gemma-3-4b ≈ nemotron-9b,
   gemma-3-12b > gpt-oss-20b, gemma-4-31b > everything.
2. **Dense beats low-active MoE at similar totals** here: gemma-4-31b
   (dense) > gemma-4-26b (4B active) on T1 by 0.05; qwen3-30b-a3b (3B
   active) ≈ qwen3-8b (dense). Active parameters, not totals, predict
   Task 1 quality.
3. **Task 2 segment-count discipline scales with model size/reasoning**:
   length-mismatch rate 41% (4B) → 19% (9B) → 15% (12B) → 1.6%
   (gpt-oss-20b). Queued v1.1 fix: pass segments as explicitly numbered
   strings; re-run only the finalists' T2 rows after it.
4. **On OpenRouter, availability = provider redundancy, not price.**
   Single-upstream models starve even paid (granite/WandB,
   llama-3.2-3b/Venice); multi-provider models never blocked. Treat
   provider count as a D10 tie-breaker henceforth.
5. **CO is the floor label for every model** (F1 0.15–0.33; the κ=0.114
   ancestor label) while ST recall is near ceiling for most — so
   optimization pressure should target CO precision policy (GEPA
   feedback) and ST precision (D11 thresholds), not rare-label recall.
6. **Reasoning-model pathology observed** (gpt-oss-20b): degenerate
   reasoning loop on an ST-dense editorial, burning all 4,000 tokens →
   null output. Rare (1/182) but argues for treating null responses as
   retriable and keeping `max_tokens` generous.
7. Latest-generation entrants (qwen3.6-27b, ministral-14b-2512) are
   competitive but do not overturn Gemma-4, and both carry unverified
   weight licenses for the closed track.

## Cost / quota ledger

≈ $3.82 key spend this week = the entire bake-off (all paid runs incl.
thinking-model output inflation) + ~750 free-tier requests (9B/20B/smoke).
Account balance remaining ≈ $9.6. A GEPA-light compile on gemma-4-31b
paid ≈ $2–4 (estimate from these runs) if the `:free` route stays
contended — affordable but worth an off-peak `:free` retry first.

## Next steps

1. v1.1 numbered-segments prompt; re-run gemma-4-31b/26b T2 val rows.
2. First dev submission probe: gemma-4-31b zero-shot on the 217 dev
   inputs, both tasks (one file serves all three settings), package and
   upload — validates format + calibrates local-val vs leaderboard.
3. Stage 1 (BFRS, `metric_threshold=1.0`) on gemma-4-31b; runner-up
   ordering check on gemma-4-26b; official-score selection per D8.
4. Retry gemma-4 `:free` routes off-peak; if reliably alive, compiles run
   free.
