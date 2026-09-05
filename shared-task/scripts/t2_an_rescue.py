"""T2 AN-rescue span judge (registered gate in CREATIVE_HEADROOM_RESEARCH.md).

Recorded v3 spans labeled AS or CO, in paragraphs where the v2.2 T1 OOF
composition fires AN, are judged for concrete-instance form: is the
<TARGET> span an anecdote (specific event, case, personal experience,
historical example = AN) rather than the author's stance (AS) or a shared
premise (CO)? Contrastive out-of-fold exemplars (10 gold-AN span texts +
10 recorded-AS spans whose dominant gold is AS), 3 rollouts at T=0.7,
FLIP to AN only on unanimous yes among successful rollouts (min 2);
errors never flip. Fold-0 screen (>= recorded + 0.015) before folds 1-4;
gate = pooled OOF >= 0.7134 AND >=3/5 fold wins.

Example (from shared-task/):
  uv run scripts/t2_an_rescue.py --model gemma-4-31b-paid --fold 0
  uv run scripts/t2_an_rescue.py --model gemma-4-31b-paid --fold all
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dspy

from daleel.candidates import marked_context
from daleel.data import TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.models import SPECS, make_lm
from daleel.policy import LABEL_POLICY
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold

ROLE_RUNS = [
    "20260710-193247-709106Z-v3-role-gepa-gemma-4-31b-paid-both-outer0-inner0-1447e7ced5",
    "20260710-210609-072093Z-v3-role-gepa-gemma-4-31b-paid-both-outer1-inner0-d9995eddba",
    "20260710-211213-843183Z-v3-role-gepa-gemma-4-31b-paid-both-outer2-inner0-1ce74f2e9a",
    "20260710-215536-635075Z-v3-role-gepa-gemma-4-31b-paid-both-outer3-inner0-30cdb1492f",
    "20260710-221215-667710Z-v3-role-gepa-gemma-4-31b-paid-both-outer4-inner0-49f2e2dad5",
]
T1_RUN = EXPERIMENTS_DIR / "20260715-003004-912987Z-t1-routed-v2-k5-n430-4f30d408fa"
SOURCE_LABELS = ("AS", "CO")
SCREEN_MARGIN = 0.015
ROLLOUTS = 3
N_EXEMPLARS = 10


class VerifyAnecdote(dspy.Signature):
    __doc__ = f"""A span extractor labeled the <TARGET> span AS or CO, but
gold-annotated anecdotes (AN) are frequently mislabeled that way. Decide
from FORM whether the target is a concrete instance: does it narrate or
cite a SPECIFIC event, incident, case, personal experience, or historical
example (= AN)? A general claim, stance, evaluation, or shared premise is
NOT an anecdote, however factual it sounds. Match the convention shown by
the labeled examples.

- {LABEL_POLICY["AN"]}"""

    examples: str = dspy.InputField(
        desc="gold-labeled example spans: anecdotes, then non-anecdote stance/premise spans")
    context: str = dspy.InputField(desc="paragraph excerpt with the <TARGET> marked")
    target: str = dspy.InputField(desc="the exact span text")
    genre: str = dspy.InputField()
    is_anecdote: bool = dspy.OutputField(
        desc="True only if the span is a concrete instance per the convention")


def load_fold(run_name: str):
    rows = list(read_jsonl(EXPERIMENTS_DIR / run_name / "predictions" / "outer_task_2.jsonl"))
    spans = {r["paragraph_id"]: [Span(s["start_offset"], s["end_offset"], s["label"])
                                 for s in r["labels"]] for r in rows}
    text = {r["paragraph_id"]: r["text"] for r in rows}
    genre = {r["paragraph_id"]: r["type"] for r in rows}
    return spans, text, genre


def dominant_gold(gold_spans, span):
    mass = {}
    for g in gold_spans:
        ov = max(0, min(g.end, span.end) - max(g.start, span.start))
        if ov > 0:
            mass[g.label] = mass.get(g.label, 0) + ov
    return max(mass, key=mass.get) if mass else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--fold", required=True,
                    help="0..4 for one fold (screen), or 'all'")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N candidate spans only (smoke test)")
    args = ap.parse_args()

    folds = range(5) if args.fold == "all" else [int(args.fold)]

    pred, text_of, genre_of, fold_of = {}, {}, {}, {}
    for f, run in enumerate(ROLE_RUNS):
        spans, text, genre = load_fold(run)
        pred.update(spans)
        text_of.update(text)
        genre_of.update(genre)
        fold_of.update({p: f for p in spans})
    gold = {p: s for p, s in clean_task2_gold(load_records(TRAIN_TASK2)).items()
            if p in pred}
    t1_an = {r["paragraph_id"] for r in read_jsonl(T1_RUN / "predictions" / "preds.jsonl")
             if "AN" in r["labels"]}

    # fold-excluded contrastive exemplars, deterministic (sorted, first N)
    gold_an_spans = sorted(
        ((p, g) for p in sorted(gold) for g in gold[p] if g.label == "AN"),
        key=lambda pg: (int(pg[0]), pg[1].start))
    true_as_spans = sorted(
        ((p, s) for p in sorted(pred) for s in pred[p]
         if s.label == "AS" and dominant_gold(gold.get(p, []), s) == "AS"),
        key=lambda ps: (int(ps[0]), ps[1].start))

    def block(items, verdict, fold):
        chosen = [(p, s) for p, s in items if fold_of[p] != fold][:N_EXEMPLARS]
        return "\n\n".join(
            f"[{verdict} — example {i + 1}]\n{text_of[p][s.start:s.end]}"
            for i, (p, s) in enumerate(chosen))

    examples_by_fold = {
        f: block(gold_an_spans, "ANECDOTE (AN)", f) + "\n\n"
           + block(true_as_spans, "NOT ANECDOTE (stance AS)", f)
        for f in range(5)
    }

    t1_an_int = {int(q) for q in t1_an}
    jobs = []  # (pid, span_index)
    for p in sorted(pred, key=int):
        if fold_of[p] not in folds or int(p) not in t1_an_int:
            continue
        for i, s in enumerate(pred[p]):
            if s.label in SOURCE_LABELS:
                jobs.append((p, i))
    if args.limit:
        jobs = jobs[: args.limit]

    spec = SPECS[args.model]
    judge = dspy.ChainOfThought(VerifyAnecdote)
    verdicts: dict[tuple, list[bool]] = {j: [] for j in jobs}
    errors: list[str] = []

    for rollout in range(ROLLOUTS):
        lm = make_lm(spec, temperature=0.7, max_tokens=args.max_tokens,
                     rollout_id=rollout)

        def judge_one(job):
            p, i = job
            s = pred[p][i]
            last = None
            for attempt in range(4):
                try:
                    with dspy.context(lm=lm):
                        result = judge(
                            examples=examples_by_fold[fold_of[p]],
                            context=marked_context(text_of[p], s.start, s.end),
                            target=text_of[p][s.start:s.end],
                            genre=genre_of[p])
                    return bool(result.is_anecdote)
                except Exception as e:
                    last = f"r{rollout} {p}:{i}: {type(e).__name__}"
                    time.sleep(min(75, 25 * (attempt + 1)))
            errors.append(last)
            return None

        with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
            for job, verdict in zip(jobs, pool.map(judge_one, jobs)):
                if verdict is not None:
                    verdicts[job].append(verdict)

    flips = {j for j, v in verdicts.items() if len(v) >= 2 and all(v)}

    relabeled = {}
    for p in pred:
        relabeled[p] = [
            Span(s.start, s.end, "AN") if (p, i) in flips else s
            for i, s in enumerate(pred[p])
        ]

    scored_pids = [p for p in pred if fold_of[p] in folds]
    g_sub = {p: gold[p] for p in scored_pids}
    rec_f1 = span_partial_f1(g_sub, {p: pred[p] for p in scored_pids})["f1"]
    new_f1 = span_partial_f1(g_sub, {p: relabeled[p] for p in scored_pids})["f1"]

    flip_true = sum(dominant_gold(gold.get(p, []), pred[p][i]) == "AN"
                    for p, i in flips)
    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md registered T2 AN-rescue",
        "model": spec.key,
        "folds": sorted(folds),
        "rollouts": ROLLOUTS,
        "temperature": 0.7,
        "source_labels": list(SOURCE_LABELS),
        "n_candidates": len(jobs),
        "n_flipped": len(flips),
        "flips_dominant_gold_AN": flip_true,
        "n_program_errors": len(errors),
        "errors_sample": errors[:5],
        "recorded_f1": round(rec_f1, 4),
        "relabeled_f1": round(new_f1, 4),
        "delta": round(new_f1 - rec_f1, 4),
    }
    if args.fold != "all":
        metrics["screen"] = {
            "rule": f"relabeled >= recorded + {SCREEN_MARGIN}",
            "pass": bool(new_f1 >= rec_f1 + SCREEN_MARGIN),
        }
    else:
        per_fold = {}
        wins = 0
        for f in range(5):
            sub = [p for p in pred if fold_of[p] == f]
            g = {p: gold[p] for p in sub}
            r0 = span_partial_f1(g, {p: pred[p] for p in sub})["f1"]
            r1 = span_partial_f1(g, {p: relabeled[p] for p in sub})["f1"]
            wins += r1 >= r0
            per_fold[str(f)] = {"recorded": round(r0, 4), "relabeled": round(r1, 4)}
        gate = round(rec_f1 + 0.02, 4)
        metrics["gate"] = {
            "rule": f"pooled >= {gate} AND >=3/5 fold wins",
            "per_fold": per_fold,
            "fold_wins": wins,
            "adopted": bool(new_f1 >= rec_f1 + 0.02 and wins >= 3),
        }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "all" if args.fold == "all" else f"f{args.fold}"
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t2-an-rescue-{spec.key}-{tag}"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "relabeled_task_2.jsonl", [
        {"paragraph_id": p, "text": text_of[p], "type": genre_of[p],
         "labels": [{"label": s.label, "start_offset": s.start,
                     "end_offset": s.end} for s in relabeled[p]]}
        for p in sorted(relabeled, key=int) if fold_of[p] in folds
    ])
    write_jsonl(pred_dir / "flip_decisions.jsonl", [
        {"paragraph_id": p, "span_index": i,
         "rollout_verdicts": verdicts[(p, i)], "flipped": (p, i) in flips}
        for p, i in jobs
    ])
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "t2_an_rescue.py", "argv": sys.argv[1:],
         "model": spec.__dict__, "t1_run": str(T1_RUN),
         "role_runs": ROLE_RUNS, "dspy_version": dspy.__version__},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
