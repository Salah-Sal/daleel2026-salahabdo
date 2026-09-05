"""Many-shot CO ranking judge (HEADROOM_AUDIT.md P1).

CO resisted definition-based judging (theory-variable crossfit 0.07), but
this corpus's conventions ARE learnable from examples (encoder AN/OT legs;
GEPA AN decision accuracy 0.74->0.88). This instrument therefore never
classifies CO — it emits a 0-10 ranking score per paragraph, intended as a
fourth rank feature for the v2 CO budget (rank-mean over encoder sigmoid,
vote fraction, span presence). The judge sees the corpus CONVENTION text
plus verbatim gold-CO exemplar paragraphs:

- train mode (default, 430 pool): exemplars for a paragraph come from the
  OTHER four folds only — honest cross-fitting;
- deploy mode (--input): exemplars are all 430-side gold-CO paragraphs
  (the target carries no gold, so nothing leaks).

The composition itself is NOT changed here: metrics.json reports the
per-fold CO-F1 of the 3-feature ranker vs +judge at the frozen k multiplier
as evidence for the next pre-registered bundle (plateau rule: adopt the
feature only if it wins or ties on >=3/5 folds).

Examples (from shared-task/):
  uv run scripts/co_judge_rank.py --model gemma-4-31b-paid \
      --encoder experiments/<quarter-oof-run> \
      --rollouts experiments/<r0> ... experiments/<r4> \
      --span-runs experiments/<v3-outer0> ... experiments/<v3-outer4>
  uv run scripts/co_judge_rank.py --model gemma-4-31b-paid \
      --input ../resources/repos/Daleel2026/data/dev/dev_in.jsonl \
      --source-id daleel2026:dev-input
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel import runtime as _runtime  # noqa: E402,F401 -- before dspy import

import dspy

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.models import SPECS, make_lm
from daleel.policy import TASK_CONTEXT
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids

from route_task1 import binary_f1, load_preds  # noqa: E402
from route_task1_v2 import CO_BUDGET_MULT, load_span_sets, rank_of  # noqa: E402
from route_task1 import CO_PRIOR  # noqa: E402

EXEMPLAR_CHARS = 350
MAX_EXEMPLARS = 20


class ScoreCommonGround(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

Rate how likely the paragraph contains at least one COMMON-GROUND unit
under THIS corpus's annotation convention.

The corpus convention is broader than the textbook definition: a
common-ground unit states shared, accepted background the audience would
not dispute — timeless common knowledge, proverbs, definitions, AND
settled legal/procedural/institutional background facts presented as
settled context. It is NOT: a specific concrete event or example (that is
an anecdote even when presented as settled), a claim the author argues
for, reported speech, or statistics.

`exemplars` contains verbatim paragraphs from this corpus that DO contain
a common-ground unit. Score the new paragraph 0-10 by how strongly it
resembles them in carrying settled shared background: 0 = certainly none,
10 = certainly contains one."""

    exemplars: str = dspy.InputField(
        desc="numbered verbatim corpus paragraphs that contain a common-ground unit"
    )
    text: str = dspy.InputField(desc="the paragraph to score")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    co_score: int = dspy.OutputField(desc="integer 0-10")


def render_exemplars(rows: list[dict]) -> str:
    parts = []
    for i, r in enumerate(rows[:MAX_EXEMPLARS], 1):
        text = r["text"][:EXEMPLAR_CHARS]
        parts.append(f"[{i}] ({r['type']}) {text}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="first N paragraphs (smoke)")
    # train-mode diagnostics inputs (optional but recommended)
    ap.add_argument("--encoder", type=Path, default=None)
    ap.add_argument("--rollouts", type=Path, nargs="*", default=[])
    ap.add_argument("--span-runs", type=Path, nargs="*", default=[])
    # deploy mode
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--source-id", default=None)
    args = ap.parse_args()
    if args.input and not args.source_id:
        raise SystemExit("--input requires --source-id")

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = {pid: labels for pid, labels in clean_task1_gold(t1, t2).items()
            if pid in set(train_ids)}
    by_id = {r["paragraph_id"]: r for r in t1}
    co_pids = sorted(pid for pid in gold if "CO" in gold[pid])

    fold_of: dict[int, int] = {}
    if args.encoder:
        for r in read_jsonl(args.encoder / "predictions" / "oof_task_1_scores.jsonl"):
            fold_of[r["paragraph_id"]] = r["fold"]

    if args.input:
        targets = read_jsonl(args.input)
        exemplar_for = {r["paragraph_id"]: [by_id[p] for p in co_pids] for r in targets}
        mode = "deploy"
    else:
        if not fold_of:
            raise SystemExit("train mode needs --encoder for the fold partition")
        targets = [by_id[pid] for pid in sorted(gold)]
        exemplar_for = {
            pid: [by_id[p] for p in co_pids if fold_of[p] != fold_of[pid]]
            for pid in sorted(gold)
        }
        mode = "train-oof"
    if args.limit:
        targets = targets[: args.limit]

    spec = SPECS[args.model]
    dspy.configure(lm=make_lm(spec, max_tokens=args.max_tokens))
    judge = dspy.Predict(ScoreCommonGround)

    def score_one(r: dict):
        pid = r["paragraph_id"]
        block = render_exemplars(exemplar_for[pid])
        last = None
        for attempt in range(4):
            try:
                pred = judge(exemplars=block, text=r["text"], genre=r["type"])
                raw = pred.co_score
                val = max(0, min(10, int(raw)))
                return pid, val, None
            except Exception as e:  # containment: unscored, never dropped
                last = f"{pid}: {type(e).__name__}: {e}"
                time.sleep(min(75, 25 * (attempt + 1)))
        return pid, None, last

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(score_one, targets))
    wall = time.time() - t0
    scores = {pid: val for pid, val, _ in results}
    errors = [err for _, _, err in results if err]

    resolved_config = {
        "script": "co_judge_rank.py",
        "argv": sys.argv[1:],
        "milestone": "HEADROOM_AUDIT.md P1",
        "mode": mode,
        "model": spec.key,
        "litellm_id": spec.litellm_id,
        "exemplar_chars": EXEMPLAR_CHARS,
        "max_exemplars": MAX_EXEMPLARS,
        "n_gold_co_exemplar_pool": len(co_pids),
        "target": (
            {"path": str(args.input), "sha256": file_sha256(args.input),
             "source_id": args.source_id}
            if args.input else "train-430"
        ),
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"co-judge-{mode}-{spec.key}-n{len(targets)}", resolved_config
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)
    write_jsonl(
        exp_dir / "predictions" / "co_judge_scores.jsonl",
        [{"paragraph_id": pid, "co_score": scores[pid]} for pid, _, _ in results],
    )

    metrics: dict = {
        "mode": mode,
        "n_targets": len(targets),
        "n_errors": len(errors),
        "errors_sample": errors[:5],
        "unscored": sum(1 for v in scores.values() if v is None),
        "wall_seconds": round(wall, 1),
        "score_histogram": {
            str(v): sum(1 for s in scores.values() if s == v) for v in range(11)
        },
    }

    # ---- train-mode diagnostics: 3-feature ranker vs +judge, per fold
    if mode == "train-oof" and args.rollouts and args.span_runs:
        rollout_preds = [load_preds(d) for d in args.rollouts]
        votes = {
            pid: {label: sum(label in p[pid] for p in rollout_preds)
                  for label in LABELS}
            for pid in scores
        }
        span = load_span_sets(list(args.span_runs))
        enc_scores = {
            r["paragraph_id"]: r["scores"]
            for r in read_jsonl(args.encoder / "predictions" / "oof_task_1_scores.jsonl")
        }
        gold_co = {pid for pid in scores if "CO" in gold[pid]}
        folds = sorted(set(fold_of[pid] for pid in scores))
        perfold = {}
        fired_ctrl: set[int] = set()
        fired_judge: set[int] = set()
        for f in folds:
            members = [pid for pid in scores if fold_of[pid] == f]
            k = max(1, round(CO_BUDGET_MULT * CO_PRIOR * len(members)))
            base_feats = [
                rank_of(members, lambda p: -enc_scores[p]["CO"]),
                rank_of(members, lambda p: (-votes[p]["CO"], -enc_scores[p]["CO"])),
                rank_of(members, lambda p: (0 if "CO" in span[p] else 1,
                                            -enc_scores[p]["CO"])),
            ]
            judge_feat = rank_of(
                members,
                lambda p: (-(scores[p] if scores[p] is not None else -1),
                           -enc_scores[p]["CO"]),
            )
            def top(feats):
                agg = {p: sum(r[p] for r in feats) / len(feats) for p in members}
                return set(sorted(members, key=lambda p: agg[p])[:k])
            ctrl, plus = top(base_feats), top(base_feats + [judge_feat])
            fired_ctrl |= ctrl
            fired_judge |= plus
            gf = {p for p in members if p in gold_co}
            perfold[str(f)] = {
                "control_3feat": round(binary_f1(ctrl, gf), 4),
                "plus_judge": round(binary_f1(plus, gf), 4),
            }
        wins = sum(v["plus_judge"] >= v["control_3feat"] for v in perfold.values())
        metrics["co_rank_diagnostics"] = {
            "k_mult": CO_BUDGET_MULT,
            "per_fold_f1": perfold,
            "pooled_control_3feat": round(binary_f1(fired_ctrl, gold_co), 4),
            "pooled_plus_judge": round(binary_f1(fired_judge, gold_co), 4),
            "judge_alone_topk": round(
                binary_f1(
                    set().union(*(
                        set(sorted(
                            [p for p in scores if fold_of[p] == f],
                            key=lambda p: (-(scores[p] if scores[p] is not None else -1),
                                           -enc_scores[p]["CO"]),
                        )[: max(1, round(CO_BUDGET_MULT * CO_PRIOR * sum(
                            1 for q in scores if fold_of[q] == f)))])
                        for f in folds
                    )),
                    gold_co,
                ), 4),
            "fold_wins_plus_judge": wins,
            "plateau_pass": wins >= 3,
        }

    atomic_write_json(exp_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nscores: {exp_dir / 'predictions' / 'co_judge_scores.jsonl'}")


if __name__ == "__main__":
    main()
