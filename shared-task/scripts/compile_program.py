"""Optimizer compile runner: D7 stages 1-3 (BFRS, MIPROv2, GEPA).

Compiles a task program with a chosen optimizer, entirely inside the
train side of the frozen D9 split: demos are bootstrapped from opt_train
(281 paragraphs) and candidates are scored on opt_val (149) — the frozen
182-paragraph val is NEVER seen by an optimizer; it stays the paired
selection set for compiled-vs-zero-shot comparisons
(`run_zero_shot.py --compiled <state.json>`).

Call budgets (survey of the 3.3.0b1 clone, 2026-07-09):
  bfrs   (num_candidate_programs+3) x len(opt_val) evals + bootstrapping
  mipro  auto=light: valset capped at 100; ~10 trials x minibatch 35
         + ~3 full evals + 6 demo-set bootstraps + ~15 proposal calls
  gepa   auto=light, 1 predictor: ~4xlen(opt_val) + 380 metric calls;
         instructions only (never touches demos); needs reflection_lm

Examples (from shared-task/):
  uv run scripts/compile_program.py --task 1 --optimizer bfrs --model gemma-4-31b-paid
  uv run scripts/compile_program.py --task 2 --optimizer gepa --model gemma-4-31b-paid

Artifacts land in experiments/<date>-t<task>-<optimizer>-<model>/:
config.json + metrics.json (committed) and compiled/*.json — program
state AND instructions dump, both GITIGNORED: bootstrapped demos embed
dataset paragraphs, and GEPA-evolved instructions can too (observed:
reflection wrote verbatim training examples into the instruction text).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dspy

from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.dspy_metrics import Task1Metric, Task2Metric
from daleel.dspy_programs import (
    QuoteProgram,
    Task1Program,
    task1_examples,
    task2_examples,
)
from daleel.models import SPECS, make_lm
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, clean_task2_gold, optimizer_split

# D8: T1 demos must be perfect (set-F1 == 1.0); T2 gold spans keep
# whitespace edges that aligned/trimmed predictions can't reproduce, so
# 1.0 is unreachable — gate at 0.9.
BOOTSTRAP_GATE = {1: 1.0, 2: 0.9}


def build_examples(task: int, opt_train_ids, opt_val_ids):
    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    by_id = {r["paragraph_id"]: r for r in t1}
    if task == 1:
        gold = clean_task1_gold(t1, t2)
        make = lambda ids: task1_examples([by_id[i] for i in ids], gold)  # noqa: E731
        gold_spans = clean_task2_gold(t2)
        texts = {r["paragraph_id"]: r["text"] for r in t1}
        metric = Task1Metric(gold_spans, texts)
    else:
        gold = clean_task2_gold(t2)
        make = lambda ids: task2_examples([by_id[i] for i in ids], gold)  # noqa: E731
        metric = Task2Metric()
    return make(opt_train_ids), make(opt_val_ids), metric


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", type=int, choices=(1, 2), required=True)
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--optimizer", choices=("bfrs", "mipro", "gepa"), required=True)
    ap.add_argument("--threads", type=int, default=8,
                    help="paid routes tolerate more than the :free 20 req/min")
    ap.add_argument("--max-tokens", type=int, default=6000,
                    help="quote runs need >=6000 (long paragraphs truncate)")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--candidates", type=int, default=8,
                    help="BFRS num_candidate_programs")
    ap.add_argument("--auto", default="light", choices=("light", "medium", "heavy"),
                    help="MIPROv2/GEPA budget preset")
    ap.add_argument("--reflection-model", default=None,
                    help="GEPA reflection LM (default: same as --model)")
    ap.add_argument("--teacher-model", default=None,
                    help="bfrs/mipro: bootstrap demos with this LM as teacher "
                    "(teacher_settings); e.g. deepseek-v4-pro — open track / "
                    "pending the Jul 13 compile-time-legality ruling")
    ap.add_argument("--prompt-model", default=None,
                    help="mipro: LM for instruction proposal (default: --model)")
    ap.add_argument("--max-metric-calls", type=int, default=None,
                    help="GEPA only: explicit budget instead of --auto "
                    "(smoke tests; auto-light on the full opt_val is ~980)")
    ap.add_argument("--opt-train-n", type=int, default=None,
                    help="cap opt_train (smoke tests), deterministic prefix")
    ap.add_argument("--opt-val-n", type=int, default=None,
                    help="cap opt_val (smoke tests), deterministic prefix")
    args = ap.parse_args()

    spec = SPECS[args.model]
    dspy.configure(lm=make_lm(spec, max_tokens=args.max_tokens))

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    opt_train_ids, opt_val_ids = optimizer_split(t1, t2)
    if args.opt_train_n:
        opt_train_ids = opt_train_ids[: args.opt_train_n]
    if args.opt_val_n:
        opt_val_ids = opt_val_ids[: args.opt_val_n]
    trainset, valset, metric = build_examples(args.task, opt_train_ids, opt_val_ids)

    # Task 1 = dedicated classifier (D2); Task 2 = QuoteProgram (D3 revised).
    program = Task1Program() if args.task == 1 else QuoteProgram()
    gate = BOOTSTRAP_GATE[args.task]

    suffix = ""
    if args.optimizer == "gepa" and args.reflection_model:
        suffix += f"-refl-{args.reflection_model}"
    if args.teacher_model:
        suffix += f"-teach-{args.teacher_model}"
    run_name = f"{time.strftime('%Y%m%d')}-t{args.task}-{args.optimizer}-{spec.key}{suffix}"
    exp_dir = EXPERIMENTS_DIR / run_name
    compiled_dir = exp_dir / "compiled"  # gitignored: demos embed dataset text
    compiled_dir.mkdir(parents=True, exist_ok=True)

    teacher_settings = (
        {"lm": make_lm(SPECS[args.teacher_model], max_tokens=args.max_tokens)}
        if args.teacher_model
        else None
    )
    t0 = time.time()
    if args.optimizer == "bfrs":
        tele = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            metric_threshold=gate,
            teacher_settings=teacher_settings,
            max_bootstrapped_demos=4,
            # labeled demos carry gold outputs without reasoning; the quote
            # program's gold examples have no `adus` field at all — disable.
            max_labeled_demos=4 if args.task == 1 else 0,
            num_candidate_programs=args.candidates,
            num_threads=args.threads,
        )
        compiled = tele.compile(program, trainset=trainset, valset=valset)
    elif args.optimizer == "mipro":
        tele = dspy.MIPROv2(
            metric=metric,
            auto=args.auto,
            metric_threshold=gate,
            teacher_settings=teacher_settings or {},
            prompt_model=make_lm(SPECS[args.prompt_model]) if args.prompt_model else None,
            max_bootstrapped_demos=4,
            max_labeled_demos=4 if args.task == 1 else 0,
            num_threads=args.threads,
            seed=args.seed,
        )
        compiled = tele.compile(program, trainset=trainset, valset=valset)
    else:  # gepa
        reflection_spec = SPECS[args.reflection_model or args.model]
        budget = (  # exactly one of auto/max_metric_calls may be passed
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

    # Stem becomes the eval run tag (run_zero_shot --compiled), so keep it
    # short; the experiment folder name already carries task and model.
    state_path = compiled_dir / f"{args.optimizer}.json"
    compiled.save(state_path, save_program=False)

    # GEPA-evolved instructions can embed verbatim training paragraphs
    # (observed 07-09: DeepSeek reflection wrote worked examples into the
    # instruction text), so instruction dumps live in the gitignored
    # compiled/ dir alongside the state; reports quote them selectively.
    instructions = {
        name: pred.signature.instructions
        for name, pred in compiled.named_predictors()
    }
    demo_counts = {
        name: len(pred.demos) for name, pred in compiled.named_predictors()
    }
    (compiled_dir / "instructions.json").write_text(
        json.dumps(instructions, indent=2, ensure_ascii=False)
    )

    config = {
        "script": "compile_program.py",
        "argv": sys.argv[1:],
        "model": spec.__dict__,
        "optimizer": args.optimizer,
        "bootstrap_gate": gate,
        "teacher_model": args.teacher_model,
        "prompt_model": args.prompt_model,
        "reflection_model": (args.reflection_model or args.model)
        if args.optimizer == "gepa" else None,
        "opt_train": len(trainset),
        "opt_val": len(valset),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "dspy_version": dspy.__version__,
    }
    metrics = {
        "task": args.task,
        "model": spec.key,
        "optimizer": args.optimizer,
        "wall_seconds": round(wall, 1),
        "demo_counts": demo_counts,
        "state_file": str(state_path),
        "instructions_changed": {
            name: instructions[name] != pred.signature.instructions
            for name, pred in program.named_predictors()
        },
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\ncompiled state: {state_path}")
    print(
        "evaluate with: uv run scripts/run_zero_shot.py "
        f"--task {args.task} --model {spec.key} --on val "
        f"{'--program quote ' if args.task == 2 else ''}--compiled {state_path}"
    )


if __name__ == "__main__":
    main()
