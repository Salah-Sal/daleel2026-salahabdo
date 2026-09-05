"""End-to-end zero-shot runner: program -> predictions -> official scores.

The stage-0 pipeline (D7). Runs the seed DSPy program for either task on
the local val split (scored against D9-cleaned gold with the official
metrics), a small train slice (plumbing smoke test), or the dev input
(produces a submission-ready task_N.jsonl). Because a zero-shot program
has no trainset dependency, one dev prediction file serves all three
training settings.

Examples (from shared-task/):
  uv run scripts/run_zero_shot.py --task 1 --model llama-3.2-3b --on train:8
  uv run scripts/run_zero_shot.py --task 1 --model nemotron-nano-9b --on val
  uv run scripts/run_zero_shot.py --task 2 --model nemotron-nano-9b --on dev

Artifacts land in experiments/<date>-t<task>-zeroshot-<model>-<on>/:
config.json + metrics.json (committed) and predictions/*.jsonl
(gitignored — contains dataset text). Bake-off gates/scores (D10): G1 =
parse-compliance rate, S1 = official score, S2 = ST/CO recall; G2 (Arabic
sanity) is a manual read of the logged samples.
"""

import argparse
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel import runtime as _runtime  # noqa: E402,F401 -- before dspy import

import dspy

from daleel.artifacts import (
    atomic_write_json,
    create_experiment_dir,
    file_sha256,
    load_artifact_lineage,
    output_record,
    write_completion_marker,
)
from daleel.constants import LABELS
from daleel.data import DEV_INPUT, TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.dspy_metrics import Task1Metric, Task2Metric, parse_label_set
from daleel.dspy_programs import (
    QuoteProgram,
    Task1Program,
    Task2Program,
    task1_examples,
    task2_examples,
)
from daleel.guideline_demos import categorize_demos, segment_demos
from daleel.guideline_programs import GuidelineProgram
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import span_partial_f1, task1_macro_f1
from daleel.models import SPECS, make_lm
from daleel.provenance import ComplianceError, build_provenance_manifest, training_ids_sha256
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import (
    clean_task1_gold,
    clean_task2_gold,
    optimizer_split,
    train_val_ids,
)
from daleel.submission import (
    validate_records_against_source,
    validate_source_records,
)


def select_records(on: str) -> tuple[list[dict], bool, Path, str]:
    """Resolve --on to (records, has_gold)."""
    t1 = load_records(TRAIN_TASK1)
    if on == "dev":
        return load_records(DEV_INPUT), False, DEV_INPUT, "daleel2026:dev-input"
    t2 = load_records(TRAIN_TASK2)
    if on == "alltrain":
        return t1, True, TRAIN_TASK1, "daleel2026:train-task-1"
    train_ids, val_ids = train_val_ids(t1, t2)
    opt_val_ids = optimizer_split(t1, t2)[1]  # judge-subset selection set
    name, _, n = on.partition(":")
    ids = {"val": val_ids, "train": train_ids, "optval": opt_val_ids}.get(name)
    if ids is None:
        raise SystemExit(
            f"--on must be val[:N], train[:N], optval[:N], alltrain, or dev, got {on!r}"
        )
    if n:
        # genre-mixed subsample, deterministic
        ids = sorted(random.Random(0).sample(list(ids), int(n)))
    keep = set(ids)
    return [r for r in t1 if r["paragraph_id"] in keep], True, TRAIN_TASK1, "daleel2026:train-task-1"


def select_input(
    on: str | None,
    input_path: Path | None,
    source_id: str | None,
) -> tuple[list[dict], bool, Path, str, str]:
    if input_path is not None:
        if source_id is None:
            raise SystemExit("--input requires --source-id")
        if not input_path.is_file():
            raise SystemExit(f"--input does not exist: {input_path}")
        records = read_jsonl(input_path)
        problems = validate_source_records(records)
        if problems:
            raise SystemExit("invalid source input:\n- " + "\n- ".join(problems[:30]))
        return records, False, input_path, source_id, input_path.stem
    resolved = on or "val"
    records, has_gold, path, known_source = select_records(resolved)
    problems = validate_source_records(records)
    if problems:
        raise SystemExit("invalid selected source:\n- " + "\n- ".join(problems[:30]))
    return records, has_gold, path, known_source, resolved.replace(":", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", type=int, choices=(1, 2), required=True)
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--track", choices=("auto", "closed", "open"), default="auto",
                    help="auto records eligible registry models as Closed, others as Open")
    source_group = ap.add_mutually_exclusive_group()
    source_group.add_argument(
        "--on", default=None, help="val[:N] | train[:N] | optval[:N] | alltrain | dev"
    )
    source_group.add_argument("--input", type=Path, default=None, help="arbitrary dev/final input")
    ap.add_argument("--source-id", default=None, help="stable source identifier required by --input")
    ap.add_argument("--threads", type=int, default=3,
                    help="keep low: OpenRouter :free allows 20 req/min")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="D13 default 0.0; self-consistency rollouts (D11) raise it")
    ap.add_argument("--rollout-id", type=int, default=None,
                    help="cache-busting rollout index for self-consistency runs; "
                    "tags the experiment dir so repeated identical configs stay distinct")
    ap.add_argument("--no-cot", action="store_true", help="Predict instead of ChainOfThought")
    ap.add_argument("--program", default="seed", choices=("seed", "quote", "guideline"),
                    help="seed = per-task seed programs; quote = QuoteProgram "
                    "(quote-then-align, one extraction serves both tasks — "
                    "running task 1 then task 2 hits the DSPy cache); "
                    "guideline = GuidelineProgram (segment-then-categorize, "
                    "official-guidelines seeds — also serves both tasks)")
    ap.add_argument("--guideline-demos", default="none", choices=("none", "en"),
                    help="guideline program only: attach the official guideline "
                    "examples as few-shot demos (en = English as published)")
    ap.add_argument("--emit-dual-spans", action="store_true",
                    help="guideline program only: emit secondary categories as "
                    "additional same-offset spans (keep off until the "
                    "overlapping-gold dual check)")
    ap.add_argument("--granularity", default="connective",
                    choices=("sentence", "clause", "connective"))
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--compiled", type=Path, default=None, metavar="STATE_JSON",
                    help="load an optimizer-compiled program state "
                    "(program.save(..., save_program=False) JSON) before "
                    "running; the run folder is tagged with the file's stem")
    ap.add_argument("--max-error-rate", type=float, default=0.0,
                    help="abort rather than emit a run above this program-error fraction")
    args = ap.parse_args()

    spec = SPECS[args.model]
    if args.threads <= 0 or args.max_tokens <= 0:
        raise SystemExit("--threads and --max-tokens must be positive")
    if not 0.0 <= args.max_error_rate <= 1.0:
        raise SystemExit("--max-error-rate must lie in [0, 1]")
    if args.input is None and args.source_id is not None:
        raise SystemExit("--source-id is only valid with --input")
    if args.program != "guideline" and (args.guideline_demos != "none" or args.emit_dual_spans):
        raise SystemExit("--guideline-demos/--emit-dual-spans require --program guideline")
    records, has_gold, source_path, source_id, source_tag = select_input(
        args.on, args.input, args.source_id
    )
    if has_gold and args.input is None:
        source_path = TRAIN_TASK1 if args.task == 1 else TRAIN_TASK2
        source_id = f"daleel2026:train-task-{args.task}"
    track = args.track
    if track == "auto":
        track = (
            "closed"
            if spec.closed_track
            and spec.license_status == "verified-open"
            and 0 < spec.params_b <= 70
            else "open"
        )
    if args.program == "quote":
        architecture = "stage0-quote-v1"
    elif args.program == "guideline":
        architecture = "stage0-guideline-v1"
    else:
        architecture = "stage0-task1-v1" if args.task == 1 else "stage0-segment-v1"
    compiled_lineage = None
    compiled_manifest = None
    if args.compiled is not None:
        if track == "closed":
            raise SystemExit(
                "closed stage-0 runs refuse --compiled because legacy compiled states do not "
                "carry complete optimizer/training lineage; use the dedicated v3 bundle path"
            )
        compiled_lineage = load_artifact_lineage(
            args.compiled,
            expected_task=args.task,
            requested_track=track,
            expected_architecture=architecture,
            kind="compiled-stage0-state",
        )
        compiled_manifest = compiled_lineage.get("manifest_data")
    training_ids = list((compiled_manifest or {}).get("training_ids", []))
    data_sources = list((compiled_manifest or {}).get("data_sources", []))
    if source_id not in data_sources:
        data_sources.append(source_id)
    auxiliary = (compiled_manifest or {}).get("auxiliary_models") or {}
    reflection = SPECS[auxiliary["reflection"]["key"]] if auxiliary.get("reflection") else None
    prompt = SPECS[auxiliary["prompt"]["key"]] if auxiliary.get("prompt") else None
    teacher = SPECS[auxiliary["teacher"]["key"]] if auxiliary.get("teacher") else None
    try:
        provenance = build_provenance_manifest(
            tasks=args.task,
            track=track,
            setting="both",
            task_model=spec,
            optimizer=(compiled_manifest or {}).get("optimizer"),
            training_ids=training_ids,
            data_sources=data_sources,
            architecture_version=architecture,
            teacher_model=teacher,
            reflection_model=reflection,
            prompt_model=prompt,
        )
    except ComplianceError as exc:
        raise SystemExit(f"run is not {track}-track compliant: {exc}") from exc
    if compiled_lineage is not None:
        provenance["upstream_artifacts"] = [
            {key: value for key, value in compiled_lineage.items() if key != "manifest_data"}
        ]
    provenance["input"] = {
        "source_id": source_id,
        "path": str(source_path),
        "sha256": file_sha256(source_path),
        "selected_ids_sha256": training_ids_sha256(
            sorted(record["paragraph_id"] for record in records)
        ),
        "record_count": len(records),
    }
    provenance["program_config"] = {
        "adapter": "ChatAdapter",
        "cot": not args.no_cot,
        "granularity": args.granularity,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "rollout_id": args.rollout_id,
    }
    if args.program == "guideline":
        provenance["program_config"]["guideline_demos"] = args.guideline_demos
        provenance["program_config"]["emit_dual_spans"] = args.emit_dual_spans
    tag = "zeroshot" if args.program == "seed" else args.program
    if args.program == "guideline" and args.guideline_demos != "none":
        tag = f"guideline-{args.guideline_demos}demos"
    if args.compiled:
        tag = args.compiled.stem
    if args.rollout_id is not None:
        tag = f"{tag}-r{args.rollout_id}"
    resolved_config = {
        "script": "run_zero_shot.py",
        "argv": sys.argv[1:],
        "model": spec.__dict__,
        "temperature": args.temperature,
        "rollout_id": args.rollout_id,
        "max_tokens": args.max_tokens,
        "dspy_version": dspy.__version__,
        "track": track,
        "architecture": architecture,
        "adapter": "ChatAdapter",
        "source_id": source_id,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "selection": source_tag,
        "max_error_rate": args.max_error_rate,
    }
    if args.program == "guideline":
        resolved_config["guideline_demos"] = args.guideline_demos
        resolved_config["emit_dual_spans"] = args.emit_dual_spans
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"t{args.task}-{tag}-{spec.key}-{source_tag}",
        resolved_config,
    )
    config_path = atomic_write_json(exp_dir / "config.json", resolved_config)
    pending_provenance_path = atomic_write_json(
        exp_dir / "provenance.pending.json", provenance
    )
    lm_kwargs = {} if args.rollout_id is None else {"rollout_id": args.rollout_id}

    def fresh_lm() -> dspy.LM:
        return make_lm(
            spec,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            **lm_kwargs,
        )

    # Global configure carries the adapter and a backstop LM; the program
    # below OWNS its own fresh instance (set in its __init__), so no module
    # shares mutable LM state with another.
    dspy.configure(lm=fresh_lm(), adapter=dspy.ChatAdapter())
    program_lm = fresh_lm()

    if args.program == "quote":
        program = QuoteProgram(cot=not args.no_cot, lm=program_lm)
    elif args.program == "guideline":
        program = GuidelineProgram(
            cot=not args.no_cot, emit_dual_spans=args.emit_dual_spans, lm=program_lm
        )
        if args.guideline_demos == "en":
            # named_predictors reaches the inner Predict under ChainOfThought
            for name, predictor in program.named_predictors():
                if name.startswith("segment"):
                    predictor.demos = segment_demos()
                elif name.startswith("categorize"):
                    predictor.demos = categorize_demos()
    elif args.task == 1:
        program = Task1Program(cot=not args.no_cot, lm=program_lm)
    else:
        program = Task2Program(
            cot=not args.no_cot, granularity=args.granularity, lm=program_lm
        )
    if args.compiled:
        program.load(args.compiled)
        program.set_lm(program_lm)  # loading state must not clobber the module LM

    def predict_one(r: dict):
        # Outer retry on top of litellm's: free-tier models rate-limit
        # upstream in bursts (observed Retry-After 12-29s), and a paragraph
        # dropped to a transient 429 would silently poison a bake-off row.
        last = None
        for attempt in range(4):
            try:
                return program(text=r["text"], genre=r["type"]), None
            except Exception as e:  # containment: never drop a row
                last = f"{r['paragraph_id']}: {type(e).__name__}: {e}"
                if not dspy.is_retryable_lm_error(e) or attempt == 3:
                    break
                time.sleep(min(75, 25 * (attempt + 1)))
        return None, last

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(predict_one, records))
    wall = time.time() - t0

    # ---- collect predictions + G1 format statistics
    out_records, errors = [], []
    n_parse_fail = n_unknown = n_mismatch = 0
    align_totals: dict[str, int] = {}
    n_para_unaligned = 0  # quote mode: paragraphs with >=1 unaligned quote
    pred_t1: dict[int, set] = {}
    pred_t2: dict[int, list] = {}
    unit_totals = {"units": 0, "multi": 0, "unlabeled": 0}
    for r, (pred, err) in zip(records, results):
        pid = r["paragraph_id"]
        echo = {"paragraph_id": pid, "text": r["text"], "type": r["type"]}
        if err:
            errors.append(err)
        stats = getattr(pred, "align_stats", None) if pred else None
        if stats:
            for k, v in stats.items():
                align_totals[k] = align_totals.get(k, 0) + v
            n_para_unaligned += stats["unaligned"] > 0
        if pred is not None:
            n_mismatch += bool(getattr(pred, "length_mismatch", False))
            unit_totals["units"] += getattr(pred, "n_units", 0)
            unit_totals["multi"] += getattr(pred, "n_multi", 0)
            unit_totals["unlabeled"] += getattr(pred, "n_unlabeled", 0)
        if args.task == 1:
            labels, parsed, unknown = parse_label_set(pred) if pred else (set(), False, [])
            n_parse_fail += not parsed
            n_unknown += bool(unknown) or bool(getattr(pred, "unknown_labels", []) if pred else [])
            pred_t1[pid] = labels
            out_records.append(echo | {"labels": sorted(labels)})
        else:
            spans = list(getattr(pred, "spans", []) or [])
            n_parse_fail += pred is None
            pred_t2[pid] = spans
            out_records.append(
                echo
                | {
                    "labels": [
                        {"label": s.label, "start_offset": s.start, "end_offset": s.end}
                        for s in spans
                    ]
                }
            )

    n = len(records)
    # Task 2 format blemish: wrong label count (seed/guideline programs) or a
    # quote that could not be aligned back to the paragraph (quote/guideline).
    if args.program == "quote":
        n_blemish = n_para_unaligned
    elif args.program == "guideline":
        n_blemish = n_para_unaligned + n_mismatch
    else:
        n_blemish = n_mismatch
    g1 = 1 - (n_parse_fail + (n_blemish if args.task == 2 else 0)) / n
    metrics: dict = {
        "task": args.task,
        "model": spec.key,
        "program": args.program,
        "compiled": str(args.compiled) if args.compiled else None,
        "litellm_id": spec.litellm_id,
        "on": args.on,
        "n_paragraphs": n,
        "cot": not args.no_cot,
        "g1_format_compliance": round(g1, 4),
        "n_parse_failures": n_parse_fail,
        "n_unknown_label_outputs": n_unknown,
        "n_program_errors": len(errors),
        "wall_seconds": round(wall, 1),
        "errors_sample": errors[:5],
    }
    if args.program in ("quote", "guideline"):
        metrics["align_stats"] = align_totals
        metrics["n_paragraphs_with_unaligned"] = n_para_unaligned
        if args.program == "guideline":
            metrics["n_length_mismatch"] = n_mismatch
            metrics["guideline_units"] = unit_totals
            metrics["guideline_demos"] = args.guideline_demos
    elif args.task == 2:
        metrics["n_length_mismatch"] = n_mismatch
        metrics["granularity"] = args.granularity

    # ---- official scoring against D9-cleaned gold
    if has_gold:
        t1_all, t2_all = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
        keep = {r["paragraph_id"] for r in records}
        if args.task == 1:
            gold = {k: v for k, v in clean_task1_gold(t1_all, t2_all).items() if k in keep}
            official = task1_macro_f1(gold, pred_t1)
            metrics["s1_official_macro_f1"] = round(official["macro_f1"], 4)
            metrics["micro_f1"] = round(official["micro_f1"], 4)
            metrics["s2_rare_recall"] = {
                label: round(official["per_label"][label]["recall"], 4) for label in ("ST", "CO")
            }
            metrics["per_label_f1"] = {
                label: round(v["f1"], 4) for label, v in official["per_label"].items()
            }
            gold_spans = {k: v for k, v in clean_task2_gold(t2_all).items() if k in keep}
            texts = {r["paragraph_id"]: r["text"] for r in records}
            metric = Task1Metric(gold_spans, texts)
            examples = task1_examples(records, gold)
        else:
            gold_spans = {k: v for k, v in clean_task2_gold(t2_all).items() if k in keep}
            official = span_partial_f1(gold_spans, pred_t2)
            metrics["s1_official_span_f1"] = round(official["f1"], 4)
            metrics["precision"] = round(official["precision"], 4)
            metrics["recall"] = round(official["recall"], 4)
            metrics["per_label_f1"] = {
                label: round(v["f1"], 4) for label, v in official["per_label"].items()
            }
            metric = Task2Metric()
            examples = task2_examples(records, gold_spans)
        proxy_scores = [
            metric(ex, pred) if pred else 0.0
            for ex, (pred, _) in zip(examples, results)
        ]
        metrics["proxy_mean"] = round(sum(proxy_scores) / n, 4)

    if n and len(errors) / n > args.max_error_rate:
        failure_path = atomic_write_json(
            exp_dir / "FAILED.json",
            {
                "stage": "stage0-inference",
                "error_count": len(errors),
                "record_count": n,
                "allowed_rate": args.max_error_rate,
                "errors_sample": errors[:10],
            },
        )
        raise SystemExit(f"stage-0 program error-rate guard failed: {failure_path}")

    # ---- artifacts
    pred_dir = exp_dir / "predictions"

    task_name = f"task_{args.task}"
    problems = validate_records_against_source(out_records, records, task_name)
    if problems:
        failure_path = atomic_write_json(
            exp_dir / "FAILED.json",
            {"stage": "source-preflight", "errors": problems[:30]},
        )
        raise SystemExit(f"source-bound output preflight failed: {failure_path}")

    filename = f"task_{args.task}.jsonl" if not has_gold else "preds.jsonl"
    write_jsonl(pred_dir / filename, out_records)
    metrics_path = atomic_write_json(exp_dir / "metrics.json", metrics)
    provenance["outputs"] = [
        output_record(
            exp_dir,
            pred_dir / filename,
            kind="quote-proposals" if args.program == "quote" else f"task-{args.task}-predictions",
        )
    ]
    provenance_path = atomic_write_json(exp_dir / "provenance.json", provenance)
    pending_provenance_path.unlink(missing_ok=True)
    write_completion_marker(
        exp_dir,
        [config_path, metrics_path, provenance_path, pred_dir / filename],
        metadata={"architecture": architecture, "task": args.task, "source_id": source_id},
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\npredictions: {pred_dir / filename}")
    if not has_gold:
        print(
            "package with: uv run scripts/package_submission.py "
            f"{pred_dir / filename} --source {source_path} "
            f"--task task_{args.task} --setting <setting>"
        )


if __name__ == "__main__":
    main()
