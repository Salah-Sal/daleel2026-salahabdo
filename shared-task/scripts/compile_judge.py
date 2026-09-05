"""Decision-level compile of the Task 1 judge (v2 propose-then-verify).

The 2026-07-10 stage tests showed zero-shot same-model verification is
flat: the model can verify criteria (ST/OT) but not corpus conventions
(CO/AN). This runner attacks the conventions gap with decision-level
supervision: every (paragraph, label) pair on the optimizer split becomes
one binary example — ~1,100 opt-train decisions from 281 paragraphs —
so demos of gold CO/AN decisions are mintable (impossible at paragraph
level, where the demo gate never admitted a rare-label example) and the
metric decomposes exactly per example (the property whose absence broke
D7 optimizer-internal selection).

Compiles VerifyOne; the saved state loads directly into JudgeStage
(shared `verify` attribute). Frozen val is never touched — selection of
the judge-label subset happens on opt_val proposals, and the chosen
configuration gets ONE confirmatory frozen-val run.

Examples (from shared-task/):
  uv run scripts/compile_judge.py --optimizer gepa --model gemma-4-31b-paid \
      --reflection-model deepseek-v4-pro
  uv run scripts/compile_judge.py --optimizer bfrs --model gemma-4-31b-paid

Artifacts land in experiments/<date>-t1-judge-<optimizer>-<model>[<suffix>]/:
config.json + metrics.json (committed) and compiled/*.json (GITIGNORED —
demos and evolved instructions embed dataset text).
"""

import argparse
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dspy

from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.dspy_metrics import JudgeMetric
from daleel.dspy_programs import VerifyOne
from daleel.models import SPECS, make_lm
from daleel.policy import LABEL_POLICY, SPURIOUS_HINTS
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, clean_task2_gold, optimizer_split

JUDGED_LABELS = ("CO", "ST", "OT", "AN")


def judge_examples(ids, by_id, gold, pos_cap, neg_cap, seed) -> list[dspy.Example]:
    """Balanced per-label decision examples: all positives up to pos_cap,
    sampled negatives up to neg_cap. Deterministic."""
    rng = random.Random(seed)
    examples = []
    for label in JUDGED_LABELS:
        pos = [i for i in ids if label in gold[i]]
        neg = [i for i in ids if label not in gold[i]]
        rng.shuffle(pos)
        rng.shuffle(neg)
        for pid in pos[:pos_cap] + neg[:neg_cap]:
            r = by_id[pid]
            examples.append(
                dspy.Example(
                    paragraph_id=pid,
                    text=r["text"],
                    genre=r["type"],
                    label=label,
                    definition=LABEL_POLICY[label],
                    caution=SPURIOUS_HINTS[label],
                    present=label in gold[pid],
                ).with_inputs("text", "genre", "label", "definition", "caution")
            )
    rng.shuffle(examples)
    return examples


def accuracy_by_label(program, examples, metric, threads) -> dict:
    """Threaded eval; returns overall + per-label decision accuracy."""

    def score_one(ex):
        try:
            return metric(ex, program(**ex.inputs()))
        except Exception:
            return 0.0

    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        scores = list(pool.map(score_one, examples))
    per_label: dict[str, list[float]] = {}
    for ex, s in zip(examples, scores):
        per_label.setdefault(ex.label, []).append(s)
    out = {
        label: round(sum(v) / len(v), 4) for label, v in sorted(per_label.items())
    }
    out["overall"] = round(sum(scores) / len(scores), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--optimizer", choices=("bfrs", "gepa"), required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="judge CoT is short; 4000 is generous")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--candidates", type=int, default=8,
                    help="BFRS num_candidate_programs")
    ap.add_argument("--auto", default="light", choices=("light", "medium", "heavy"))
    ap.add_argument("--reflection-model", default=None,
                    help="GEPA reflection LM (default: same as --model)")
    ap.add_argument("--teacher-model", default=None,
                    help="BFRS: bootstrap demos with this LM as teacher")
    ap.add_argument("--max-metric-calls", type=int, default=None,
                    help="GEPA only: explicit budget (smoke tests)")
    ap.add_argument("--train-pos", type=int, default=40,
                    help="opt-train positives per label")
    ap.add_argument("--train-neg", type=int, default=40)
    ap.add_argument("--val-pos", type=int, default=25,
                    help="opt-val positives per label")
    ap.add_argument("--val-neg", type=int, default=25)
    args = ap.parse_args()

    spec = SPECS[args.model]
    dspy.configure(lm=make_lm(spec, max_tokens=args.max_tokens))

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    by_id = {r["paragraph_id"]: r for r in t1}
    gold = clean_task1_gold(t1, t2)
    opt_train_ids, opt_val_ids = optimizer_split(t1, t2)
    trainset = judge_examples(
        opt_train_ids, by_id, gold, args.train_pos, args.train_neg, args.seed
    )
    valset = judge_examples(
        opt_val_ids, by_id, gold, args.val_pos, args.val_neg, args.seed + 1
    )
    metric = JudgeMetric(
        clean_task2_gold(t2), {r["paragraph_id"]: r["text"] for r in t1}
    )

    program = VerifyOne()

    suffix = ""
    if args.optimizer == "gepa" and args.reflection_model:
        suffix += f"-refl-{args.reflection_model}"
    if args.teacher_model:
        suffix += f"-teach-{args.teacher_model}"
    run_name = f"{time.strftime('%Y%m%d')}-t1-judge-{args.optimizer}-{spec.key}{suffix}"
    exp_dir = EXPERIMENTS_DIR / run_name
    compiled_dir = exp_dir / "compiled"  # gitignored: demos embed dataset text
    compiled_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.optimizer == "bfrs":
        tele = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            metric_threshold=1.0,  # binary decisions: demos must be correct
            teacher_settings=(
                {"lm": make_lm(SPECS[args.teacher_model], max_tokens=args.max_tokens)}
                if args.teacher_model
                else None
            ),
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=args.candidates,
            num_threads=args.threads,
        )
        compiled = tele.compile(program, trainset=trainset, valset=valset)
    else:  # gepa
        reflection_spec = SPECS[args.reflection_model or args.model]
        budget = (
            {"max_metric_calls": args.max_metric_calls}
            if args.max_metric_calls
            else {"auto": args.auto}
        )
        tele = dspy.GEPA(
            metric=metric,
            reflection_lm=make_lm(reflection_spec, temperature=1.0, max_tokens=16000),
            num_threads=args.threads,
            track_stats=True,
            add_format_failure_as_feedback=True,
            log_dir=str(exp_dir / "gepa_logs"),
            seed=args.seed,
            **budget,
        )
        compiled = tele.compile(program, trainset=trainset, valset=valset)
    wall = time.time() - t0

    state_path = compiled_dir / f"judge-{args.optimizer}.json"
    compiled.save(state_path, save_program=False)
    (compiled_dir / "instructions.json").write_text(
        json.dumps(
            {n: p.signature.instructions for n, p in compiled.named_predictors()},
            indent=2, ensure_ascii=False,
        )
    )

    # decision-level report card: seed vs compiled on the opt-val decisions
    # (the subset-selection signal; frozen val stays untouched)
    seed_acc = accuracy_by_label(VerifyOne(), valset, metric, args.threads)
    compiled_acc = accuracy_by_label(compiled, valset, metric, args.threads)

    config = {
        "script": "compile_judge.py",
        "argv": sys.argv[1:],
        "model": spec.__dict__,
        "optimizer": args.optimizer,
        "reflection_model": (args.reflection_model or args.model)
        if args.optimizer == "gepa" else None,
        "teacher_model": args.teacher_model,
        "judged_labels": JUDGED_LABELS,
        "n_train_decisions": len(trainset),
        "n_val_decisions": len(valset),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "dspy_version": dspy.__version__,
    }
    metrics = {
        "stage": "judge-compile",
        "model": spec.key,
        "optimizer": args.optimizer,
        "wall_seconds": round(wall, 1),
        "demo_counts": {n: len(p.demos) for n, p in compiled.named_predictors()},
        "instructions_changed": {
            n: p.signature.instructions
            != dict(VerifyOne().named_predictors())[n].signature.instructions
            for n, p in compiled.named_predictors()
        },
        "seed_val_accuracy": seed_acc,
        "compiled_val_accuracy": compiled_acc,
        "state_file": str(state_path),
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\ncompiled state: {state_path}")


if __name__ == "__main__":
    main()
