# D3 revision — quote-then-align vs segment-then-classify (2026-07-09)

**Decision: `QuoteProgram` (quote-then-align) is the Task 2 program;
segment-then-classify becomes the ablation baseline / fallback. Task 1
keeps its dedicated multi-label classifier.** Motivation (user, 07-09):
the rule segmenter is deliberately DALEEL-tailored (connective inventory,
speaker-marker protection tuned on this corpus) and must not be presented
as a generally valid Arabic discourse segmenter; asking the model for
verbatim ADU quotes and recovering offsets programmatically removes the
hand-tuned component — and it also scores better.

## Feasibility (measured on gold before any LLM call)

- **Aligner oracle ceiling 0.999** (perfect quoter + `daleel.align`,
  cleaned training gold) vs the connective segmenter's 0.920 — the
  ceiling moves from the segmenter into the model's quoting fidelity.
- Only **0.7%** of gold quotes (21/2,973) occur more than once in their
  paragraph; a reading-order cursor with global fallback resolves them.
- Gold is not a partition: 7 paragraphs have overlapping spans, 4.86
  spans/paragraph on average (max 60), covering 83% of text.

## Implementation

- `daleel.align`: staged matching — exact (cursor) → whitespace-flexible
  → Arabic-normalized (diacritics/tatweel/alef/ya/ta-marbuta, offset map)
  → fuzzy (difflib, ratio ≥ 0.70) — then de-overlap (trim, never the
  recall>1 metric quirk). Per-stage counts are run telemetry.
- `QuoteProgram` (`daleel.dspy_programs`): signature outputs
  `adus: list[ADUQuote]` (pydantic). **Footgun:** a `Literal` label
  nested inside a pydantic item is validated case-sensitively with none
  of DSPy's top-level-Literal leniency — one lowercase label would fail
  the whole paragraph with `AdapterParseError`. `label` is therefore
  `str`, normalized and filtered program-side.
- DSPy 3.3.0b1 verified in the clone: `list[BaseModel]` outputs parse via
  `json_repair` + `TypeAdapter`, ChatAdapter auto-falls-back to
  JSONAdapter; no built-in substring/citation primitive fits (the
  experimental `Citations` type is Anthropic-native-only). `dspy.BestOfN`
  with a substring reward is the fidelity lever if ever needed.
- Runner: `run_zero_shot.py --program quote`; one extraction serves both
  tasks, so scoring Task 1 after Task 2 is a 100% DSPy-cache hit
  (observed: 1.7 s, $0).

## Results (frozen 182-para val, zero-shot seeds, temp 0)

| run | program | official F1 | G1 | note |
|---|---|---|---|---|
| T2 gemma-4-31b | **quote** | **0.6805** | 1.00 | P 0.636 / R 0.732 |
| T2 gemma-4-31b | segment | 0.6542 | 0.98 | P 0.634 / R 0.675 |
| T2 gemma-4-26b | quote | 0.6762 | 1.00 | ordering check passed |
| T2 gemma-4-26b | segment | 0.663 | 0.96 | bake-off row |
| T1 gemma-4-31b | classifier | **0.7022** | 1.00 | dedicated program stays |
| T1 gemma-4-31b | quote-derived | 0.6641 | 1.00 | unification rejected |

- Quote beats segment by **+0.026**, above the pre-registered 0.02
  actionable bar (paired on the same val set).
- Per-label deltas (quote − segment): **OT +0.149** (0.480→0.629 — the
  segmenter's connective cuts were tuned for OT and the model's own
  boundaries still beat them), AN +0.038, TE +0.021, AS +0.004,
  ST −0.029, CO −0.033 (both rare-label dips within noise; D8/D11
  optimizer pressure targets ST/CO anyway).
- Quoting fidelity (710 val quotes): 694 exact, 7 whitespace-flexible,
  5 Arabic-normalized, 4 fuzzy, **0 unaligned, 0 dropped overlaps** —
  fidelity is not the bottleneck at 31B.
- Task 1 via extraction loses 0.038 macro — extraction is a harder task
  than classification; label-set precision suffers. Task 1 program
  unchanged (D2).
- **Ordering check:** quote-then-align lifts both finalists (26b
  0.663→0.676, 31b 0.654→0.681) and 31b stays ahead — the D10 model
  decision is unaffected by the program change. Fidelity scales with
  size: 26b needed 21 fuzzy matches and 35 overlap-trims vs 31b's 4
  and 0 (same pattern as the bake-off's segment-count finding).

## Caveats

- One val paragraph hit the 6,000-token cap (truncation warning);
  `json_repair` salvaged a valid prefix. Worst under-prediction
  (paragraph 986: 23 predicted vs 46 gold spans) is *merging*, not
  truncation — its spans still cover 99% of the text and pair-sum
  scoring is tolerant of merged same-label spans. Use
  `--max-tokens 6000` (or more) for all quote runs.
- The whitespace-trim note from `Task2Metric` still applies: gold spans
  keep whitespace edges, aligned quotes are trimmed, so
  `metric_threshold=1.0` bootstrap gates stay unreachable for T2 — gate
  at ~0.9.
- The v1.1 numbered-segments fix for the segment program is moot for the
  adopted program (kept in notes for the ablation row only).

## Cost

≈ $0.11 total: 31b smoke + full val + cached T1 rescore ≈ $0.06, 26b
ordering check ≈ $0.05. Key spend this week after all runs: $3.93.

## Next steps

1. Dev submission probe with the revised programs (T1 classifier,
   T2 quote) — package + upload, calibrate local-val vs leaderboard.
2. Stage 1 BFRS on gemma-4-31b: T1 classifier + T2 QuoteProgram
   (bootstrap gate ~0.9 for T2), gemma-4-26b ordering check.
3. If a weaker/free route ever degrades quote fidelity: `dspy.BestOfN`
   with an all-quotes-align reward before falling back to the segmenter.
