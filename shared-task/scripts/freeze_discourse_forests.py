"""Freeze label-blind discourse forests over quote-proposal atoms.

The parser sees exact atom IDs/offsets/text but never draft or gold Daleel
roles.  Every model output is converted to a canonical acyclic forest and
bound to the source, proposal artifact, atomization settings, parser contract,
and executable provenance before structural role experiments may consume it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel import runtime as _runtime  # noqa: E402,F401 -- before dspy import

import dspy  # noqa: E402

from daleel.artifacts import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    create_experiment_dir,
    file_sha256,
    load_artifact_lineage,
    output_record,
    write_completion_marker,
)
from daleel.candidates import atomize_spans  # noqa: E402
from daleel.data import TRAIN_TASK1, load_records  # noqa: E402
from daleel.discourse_forest import (  # noqa: E402
    FOREST_ARCHITECTURE,
    RELATIONS,
    DiscourseForest,
    atom_inventory,
    build_forest_artifact,
    forest_nodes,
    forest_record,
    validate_forest_edges,
)
from daleel.dspy_discourse import DiscourseForestParser, ParseAtomForest  # noqa: E402
from daleel.io import read_jsonl  # noqa: E402
from daleel.metrics import Span  # noqa: E402
from daleel.models import SPECS, make_lm  # noqa: E402
from daleel.provenance import build_provenance_manifest  # noqa: E402
from daleel.runtime import EXPERIMENTS_DIR  # noqa: E402
from daleel.splits import train_val_ids  # noqa: E402
from daleel.submission import validate_records_against_source  # noqa: E402


def _load_proposals(path: Path, records: list[dict]) -> dict[int, list[Span]]:
    wanted = {record["paragraph_id"] for record in records}
    rows = [row for row in read_jsonl(path) if row.get("paragraph_id") in wanted]
    problems = validate_records_against_source(rows, records, "task_2")
    if problems:
        raise SystemExit("invalid/missing forest proposals:\n- " + "\n- ".join(problems[:30]))
    return {
        row["paragraph_id"]: [
            Span(item["start_offset"], item["end_offset"], item["label"])
            for item in row["labels"]
        ]
        for row in rows
    }


def _parser_contract_hash() -> str:
    return canonical_json_sha256(
        {
            "instructions": ParseAtomForest.instructions,
            "relations": list(RELATIONS),
            "input_fields": ["text", "genre", "atom_inventory"],
            "output_field": "edges",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--track", choices=("closed", "open"), default="closed")
    parser.add_argument("--setting", choices=("editorial", "debate", "both"), default="both")
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--pool", choices=("legacy-train", "all"), default="legacy-train")
    parser.add_argument("--granularity", choices=("sentence", "clause", "connective"),
                        default="connective")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--parser-cot", action="store_true")
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.threads <= 0 or args.max_tokens <= 0:
        raise SystemExit("--threads and --max-tokens must be positive")
    if not 0.0 <= args.max_error_rate <= 1.0:
        raise SystemExit("--max-error-rate must lie in [0, 1]")

    all_records = load_records(TRAIN_TASK1)
    if args.pool == "legacy-train":
        # train_val_ids needs Task 2 only to derive the same cleaned split.
        from daleel.data import TRAIN_TASK2  # noqa: PLC0415

        pool_ids = set(train_val_ids(all_records, load_records(TRAIN_TASK2))[0])
    else:
        pool_ids = {record["paragraph_id"] for record in all_records}
    records = [
        record for record in all_records
        if record["paragraph_id"] in pool_ids
        and (args.setting == "both" or record["type"] == args.setting)
    ]
    proposal_lineage = load_artifact_lineage(
        args.proposals,
        expected_task=2,
        requested_track=args.track,
        expected_architecture="stage0-quote-v1",
        expected_model=args.model,
        expected_output_kind="quote-proposals",
        kind="forest-quote-proposals",
    )
    proposals = _load_proposals(args.proposals, records)

    nodes_by_id = {}
    parse_examples: list[dspy.Example] = []
    deterministic_ids: set[int] = set()
    for record in records:
        pid = record["paragraph_id"]
        atoms = atomize_spans(record["text"], proposals[pid], args.granularity)
        nodes = forest_nodes(atoms)
        nodes_by_id[pid] = nodes
        if len(nodes) <= 1:
            deterministic_ids.add(pid)
            continue
        parse_examples.append(
            dspy.Example(
                paragraph_id=pid,
                text=record["text"],
                genre=record["type"],
                atom_inventory=atom_inventory(nodes),
            ).with_inputs("text", "genre", "atom_inventory")
        )

    contract = {
        "architecture": FOREST_ARCHITECTURE,
        "model": args.model,
        "track": args.track,
        "setting": args.setting,
        "pool": args.pool,
        "source_id": "daleel2026:train-task-1",
        "source_sha256": file_sha256(TRAIN_TASK1),
        "proposal_sha256": proposal_lineage["sha256"],
        "granularity": args.granularity,
        "parser_cot": args.parser_cot,
        "parser_signature_sha256": _parser_contract_hash(),
        "relation_inventory": list(RELATIONS),
        "max_tokens": args.max_tokens,
    }
    preflight = {
        "preflight_ok": True,
        "contract": contract,
        "contract_sha256": canonical_json_sha256(contract),
        "counts": {
            "paragraphs": len(records),
            "atoms": sum(len(nodes) for nodes in nodes_by_id.values()),
            "planned_parser_calls": len(parse_examples),
            "deterministic_zero_or_one_atom": len(deterministic_ids),
        },
        "proposal_lineage": {
            key: value for key, value in proposal_lineage.items() if key != "manifest_data"
        },
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"discourse-forests-{args.model}-{args.setting}-{args.pool}",
        contract,
    )
    config_path = atomic_write_json(
        exp_dir / "config.json",
        {"script": "freeze_discourse_forests.py", "argv": sys.argv[1:], **preflight},
    )
    provenance = build_provenance_manifest(
        tasks=(1, 2),
        track=args.track,
        setting=args.setting,
        task_model=SPECS[args.model],
        optimizer=None,
        training_ids=[],
        data_sources=("daleel2026:train-task-1", "daleel2026:train-task-2"),
        architecture_version=FOREST_ARCHITECTURE,
    )
    provenance["upstream_artifacts"] = [preflight["proposal_lineage"]]
    provenance["data_artifacts"] = [
        {
            "source_id": "daleel2026:train-task-1",
            "path": str(TRAIN_TASK1),
            "sha256": file_sha256(TRAIN_TASK1),
        }
    ]
    pending_path = atomic_write_json(exp_dir / "provenance.pending.json", provenance)

    lm = make_lm(SPECS[args.model], max_tokens=args.max_tokens)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    forest_parser = DiscourseForestParser(cot=args.parser_cot)
    started = time.time()
    if parse_examples:
        predictions, failed, exceptions = forest_parser.batch(
            parse_examples,
            num_threads=args.threads,
            max_errors=max(10, len(parse_examples) + 1),
            return_failed_examples=True,
            provide_traceback=False,
            timeout=0,
        )
    else:
        predictions, failed, exceptions = [], [], []
    elapsed = time.time() - started

    prediction_by_id = {
        example.paragraph_id: prediction
        for example, prediction in zip(parse_examples, predictions)
    }
    forest_rows = []
    totals: Counter[str] = Counter()
    parser_errors = 0
    for record in records:
        pid = record["paragraph_id"]
        nodes = nodes_by_id[pid]
        if pid in deterministic_ids:
            raw_edges = []
            status = "deterministic"
            error_text = None
        else:
            prediction = prediction_by_id.get(pid)
            raw_edges = getattr(prediction, "edges", None)
            if not isinstance(raw_edges, list):
                raw_edges = []
                status = "error_fallback"
                error_text = f"missing/list-invalid edges: {prediction!r}"[:500]
                parser_errors += 1
            else:
                status = "ok"
                error_text = None
        edges, validation = validate_forest_edges(nodes, raw_edges)
        if validation["dropped_edges"]:
            status = "validated_repair" if status == "ok" else status
        totals.update(
            {
                "paragraphs": 1,
                "nodes": len(nodes),
                "raw_edges": validation["raw_edges"],
                "accepted_edges": validation["accepted_edges"],
                "dropped_edges": validation["dropped_edges"],
                "roots": validation["roots"],
                f"status_{status}": 1,
            }
        )
        forest_rows.append(
            forest_record(
                DiscourseForest(pid, nodes, edges),
                validation=validation,
                parser_status=status,
                parser_error=error_text,
            )
        )

    parser_errors = max(parser_errors, len(failed))
    if records and parser_errors / len(records) > args.max_error_rate:
        failure_path = atomic_write_json(
            exp_dir / "FAILED.json",
            {
                "stage": "forest-parser",
                "parser_errors": parser_errors,
                "paragraphs": len(records),
                "allowed_rate": args.max_error_rate,
                "failure_sample": [f"{type(exc).__name__}: {exc}" for exc in exceptions[:5]],
            },
        )
        raise SystemExit(f"forest parser error-rate guard failed: {failure_path}")

    telemetry = {
        **dict(totals),
        "parser_calls": len(parse_examples),
        "parser_errors": parser_errors,
        "program_failed_examples": len(failed),
        "failure_sample": [f"{type(exc).__name__}: {exc}" for exc in exceptions[:5]],
        "wall_seconds": round(elapsed, 2),
    }
    artifact = build_forest_artifact(
        contract=contract,
        records=forest_rows,
        telemetry=telemetry,
    )
    artifact_path = atomic_write_json(exp_dir / "predictions" / "forests.json", artifact)
    metrics_path = atomic_write_json(exp_dir / "metrics.json", telemetry)
    provenance["outputs"] = [
        output_record(exp_dir, artifact_path, kind="discourse-forests"),
    ]
    provenance_path = atomic_write_json(exp_dir / "provenance.json", provenance)
    pending_path.unlink(missing_ok=True)
    write_completion_marker(
        exp_dir,
        [config_path, metrics_path, artifact_path, provenance_path],
        metadata={"architecture": FOREST_ARCHITECTURE},
    )
    print(json.dumps(telemetry, ensure_ascii=False, indent=2))
    print(f"\nfrozen forests: {artifact_path}")


if __name__ == "__main__":
    main()
