"""Run the release-safe v3 proposal -> atom -> role pipeline.

One set of atom-role calls serves Task 2 spans and Task 1 span evidence.  A
hash-bound direct Task 1 view may be fused deterministically.  Every static
input is validated before an LM is constructed; completed runs contain exact
source-bound outputs, provenance, and a final completion marker.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel import runtime as _runtime  # noqa: E402,F401 -- before dspy import

import dspy  # noqa: E402

from daleel.artifacts import (  # noqa: E402
    atomic_write_json,
    create_experiment_dir,
    file_sha256,
    load_artifact_lineage,
    output_record,
    write_completion_marker,
)
from daleel.constants import LABELS  # noqa: E402
from daleel.data import DEV_INPUT, TRAIN_TASK1, TRAIN_TASK2, load_records  # noqa: E402
from daleel.dspy_programs import QuoteProgram  # noqa: E402
from daleel.dspy_span_roles import (  # noqa: E402
    BalancedRoleDemoSelector,
    SpanRoleDecision,
    assemble_example_predictions,
    role_diagnostics,
    span_role_examples,
)
from daleel.io import read_jsonl, write_jsonl  # noqa: E402
from daleel.metrics import Span, span_partial_f1, task1_macro_f1  # noqa: E402
from daleel.models import SPECS, make_lm  # noqa: E402
from daleel.provenance import (  # noqa: E402
    ComplianceError,
    build_provenance_manifest,
    training_ids_sha256,
)
from daleel.role_artifacts import load_role_bundle  # noqa: E402
from daleel.runtime import EXPERIMENTS_DIR  # noqa: E402
from daleel.splits import (  # noqa: E402
    clean_task1_gold,
    clean_task2_gold,
    optimizer_split,
    train_val_ids,
)
from daleel.submission import (  # noqa: E402
    validate_records_against_source,
    validate_source_records,
)


ARCHITECTURE_VERSION = "proposal-atom-role-v3"
STAGE0_QUOTE_ARCHITECTURE = "stage0-quote-v1"
STAGE0_TASK1_ARCHITECTURE = "stage0-task1-v1"
FUSION_ARCHITECTURE = "task1-fusion-v1"
FUSION_OPERATIONS = frozenset({"spans", "direct", "union", "intersection"})


def _select_known_records(on: str) -> tuple[list[dict], bool, Path, str]:
    t1 = load_records(TRAIN_TASK1)
    if on == "dev":
        return load_records(DEV_INPUT), False, DEV_INPUT, "daleel2026:dev-input"
    t2 = load_records(TRAIN_TASK2)
    train_ids, val_ids = train_val_ids(t1, t2)
    opt_val_ids = optimizer_split(t1, t2)[1]
    name, separator, count_text = on.partition(":")
    ids = {"val": val_ids, "train": train_ids, "optval": opt_val_ids}.get(name)
    if ids is None:
        raise SystemExit(f"--on must be val[:N], train[:N], optval[:N], or dev; got {on!r}")
    if separator:
        try:
            count = int(count_text)
        except ValueError as exc:
            raise SystemExit(f"invalid subset count in --on {on!r}") from exc
        if not 0 < count <= len(ids):
            raise SystemExit(f"subset count must lie in [1, {len(ids)}], got {count}")
        ids = sorted(random.Random(0).sample(list(ids), count))
    keep = set(ids)
    return (
        [record for record in t1 if record["paragraph_id"] in keep],
        True,
        TRAIN_TASK1,
        "daleel2026:train-task-1",
    )


def select_records(
    *,
    on: str | None,
    input_path: Path | None,
    source_id: str | None,
) -> tuple[list[dict], bool, Path, str, str]:
    """Resolve a named split or arbitrary organizer input."""

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
    resolved_on = on or "val"
    records, has_gold, path, known_source_id = _select_known_records(resolved_on)
    problems = validate_source_records(records)
    if problems:
        raise SystemExit("invalid selected source records:\n- " + "\n- ".join(problems[:30]))
    return records, has_gold, path, known_source_id, resolved_on.replace(":", "")


def _rows_to_spans(rows: list[dict]) -> dict[int, list[Span]]:
    return {
        row["paragraph_id"]: [
            Span(item["start_offset"], item["end_offset"], item["label"])
            for item in row["labels"]
        ]
        for row in rows
    }


def load_task2_predictions(path: Path, source: list[dict]) -> dict[int, list[Span]]:
    wanted = {record["paragraph_id"] for record in source}
    rows = [row for row in read_jsonl(path) if row.get("paragraph_id") in wanted]
    problems = validate_records_against_source(rows, source, "task_2")
    if problems:
        raise SystemExit(f"invalid Task 2 predictions {path}:\n- " + "\n- ".join(problems[:30]))
    return _rows_to_spans(rows)


def load_task1_predictions(path: Path, source: list[dict]) -> dict[int, set[str]]:
    wanted = {record["paragraph_id"] for record in source}
    rows = [row for row in read_jsonl(path) if row.get("paragraph_id") in wanted]
    problems = validate_records_against_source(rows, source, "task_1")
    if problems:
        raise SystemExit(f"invalid Task 1 predictions {path}:\n- " + "\n- ".join(problems[:30]))
    return {row["paragraph_id"]: set(row["labels"]) for row in rows}


def _lineage_without_manifest(lineage: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in lineage.items() if key != "manifest_data"}


def _load_policy(path: Path) -> dict[str, str]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid fusion policy {path}: {exc}") from exc
    if not isinstance(policy, dict):
        raise SystemExit("fusion policy must be a JSON object")
    missing = sorted(set(LABELS) - set(policy))
    extra = sorted(set(policy) - set(LABELS))
    if missing or extra:
        raise SystemExit(f"fusion policy label mismatch: missing={missing}, extra={extra}")
    invalid = {label: operation for label, operation in policy.items() if operation not in FUSION_OPERATIONS}
    if invalid:
        raise SystemExit(f"fusion policy contains invalid operations: {invalid}")
    return policy


def fuse_labels(
    span_labels: set[str],
    direct_labels: set[str] | None,
    mode: str,
    per_label_policy: dict[str, str] | None = None,
) -> set[str]:
    """Deterministic two-view Task 1 reconciliation."""

    if direct_labels is None:
        if mode != "spans" or per_label_policy:
            raise ValueError("direct Task 1 predictions are required by the fusion policy")
        return set(span_labels)
    operations = {
        "spans": lambda label: label in span_labels,
        "direct": lambda label: label in direct_labels,
        "union": lambda label: label in span_labels or label in direct_labels,
        "intersection": lambda label: label in span_labels and label in direct_labels,
    }
    if mode not in operations:
        raise ValueError(f"unknown fusion mode {mode!r}")
    final: set[str] = set()
    for label in LABELS:
        operation = (per_label_policy or {}).get(label, mode)
        if operation not in operations:
            raise ValueError(f"unknown fusion operation {operation!r} for {label}")
        if operations[operation](label):
            final.add(label)
    return final


def run_quote_proposals(
    records: list[dict],
    *,
    threads: int,
    cot: bool,
) -> tuple[dict[int, list[Span]], dict[str, Any]]:
    """Run the exact stage-0 quote program with retryable-error containment."""

    program = QuoteProgram(cot=cot)

    def predict(record: dict):
        last: Exception | None = None
        for attempt in range(4):
            try:
                return program(text=record["text"], genre=record["type"]), None
            except Exception as exc:
                last = exc
                if not dspy.is_retryable_lm_error(exc) or attempt == 3:
                    break
                time.sleep(min(30, 5 * (attempt + 1)))
        return None, last

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(predict, records))

    proposals: dict[int, list[Span]] = {}
    align_totals: dict[str, int] = {}
    errors: list[str] = []
    for record, (prediction, error) in zip(records, results):
        pid = record["paragraph_id"]
        spans = list(getattr(prediction, "spans", []) or [])
        # Frozen proposals pass the source-bound preflight before atomization;
        # live spans have no such gate, and atomize_spans hard-raises on
        # out-of-range offsets. Contain a misaligned span as a recorded error
        # instead of crashing the whole run.
        in_range = []
        for span in spans:
            if 0 <= span.start < span.end <= len(record["text"]):
                in_range.append(span)
            else:
                errors.append(
                    f"{pid}: misaligned live span {span.label} "
                    f"[{span.start}, {span.end}) outside text length {len(record['text'])}"
                )
        proposals[pid] = in_range
        if error is not None:
            errors.append(f"{pid}: {type(error).__name__}: {error}")
        for key, value in (getattr(prediction, "align_stats", {}) or {}).items():
            align_totals[key] = align_totals.get(key, 0) + value
    return proposals, {"align_stats": align_totals, "errors": errors}


def _proposal_rows(records: list[dict], spans: dict[int, list[Span]]) -> list[dict]:
    return [
        {
            "paragraph_id": record["paragraph_id"],
            "text": record["text"],
            "type": record["type"],
            "labels": [
                {"label": span.label, "start_offset": span.start, "end_offset": span.end}
                for span in spans.get(record["paragraph_id"], ())
            ],
        }
        for record in records
    ]


def _resolve_bundle_setting(
    name: str,
    cli_value: Any,
    bundle_settings: dict[str, Any],
    default: Any,
) -> Any:
    recorded = bundle_settings.get(name, default)
    if cli_value is not None and cli_value != recorded:
        raise SystemExit(
            f"--{name.replace('_', '-')}={cli_value!r} does not match role bundle {recorded!r}"
        )
    return recorded


def _validate_rate(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise SystemExit(f"{name} must lie in [0, 1], got {value}")


def _error_rate_exceeded(count: int, total: int, allowed: float) -> bool:
    return bool(total) and count / total > allowed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--track", choices=("closed", "open"), default="closed")
    parser.add_argument("--setting", choices=("editorial", "debate", "both"), default="both")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--on", default=None, help="val[:N] | train[:N] | optval[:N] | dev")
    source_group.add_argument("--input", type=Path, default=None, help="arbitrary dev/final input JSONL")
    parser.add_argument("--source-id", default=None, help="stable source identifier required by --input")

    proposal_group = parser.add_mutually_exclusive_group(required=True)
    proposal_group.add_argument("--proposals", type=Path, help="frozen, hash-bound quote proposals")
    proposal_group.add_argument(
        "--live-proposals",
        action="store_true",
        help="explicitly run the stage-0 quote extractor instead of reusing an artifact",
    )
    parser.add_argument("--training-proposals", type=Path, default=None)
    parser.add_argument("--memory-pool", choices=("auto", "legacy-train", "all"), default="auto")
    parser.add_argument("--role-state", type=Path, default=None)
    parser.add_argument("--demo-selector", choices=("auto", "balanced", "none"), default="auto")
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--context-chars", type=int, default=None)
    parser.add_argument("--demo-context-chars", type=int, default=None)
    parser.add_argument("--granularity", choices=("sentence", "clause", "connective"), default=None)
    parser.add_argument("--containment-threshold", type=float, default=None)
    role_cot_group = parser.add_mutually_exclusive_group()
    role_cot_group.add_argument("--role-cot", dest="role_cot", action="store_true")
    role_cot_group.add_argument("--no-role-cot", dest="role_cot", action="store_false")
    parser.set_defaults(role_cot=None)

    parser.add_argument("--direct-task1", type=Path, default=None)
    parser.add_argument(
        "--fusion",
        choices=tuple(sorted(FUSION_OPERATIONS)),
        default="direct",
        help="direct is the Task 1 champion; spans is the self-contained ablation",
    )
    parser.add_argument("--fusion-policy", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--extractor-max-tokens", type=int, default=6000)
    parser.add_argument("--no-extractor-cot", action="store_true")
    parser.add_argument("--max-extraction-error-rate", type=float, default=0.0)
    parser.add_argument("--max-role-error-rate", type=float, default=0.0)
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate all artifacts/settings and exit before make_lm")
    args = parser.parse_args()

    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    if args.max_tokens <= 0 or args.extractor_max_tokens <= 0:
        raise SystemExit("token limits must be positive")
    _validate_rate("--max-extraction-error-rate", args.max_extraction_error_rate)
    _validate_rate("--max-role-error-rate", args.max_role_error_rate)
    if args.input is None and args.source_id is not None:
        raise SystemExit("--source-id is only valid with --input")
    if args.direct_task1 is None and (args.fusion != "spans" or args.fusion_policy):
        raise SystemExit("--direct-task1 is required by the requested Task 1 fusion")

    records, has_gold, source_path, source_id, source_tag = select_records(
        on=args.on,
        input_path=args.input,
        source_id=args.source_id,
    )
    inference_ids = {record["paragraph_id"] for record in records}
    t1_records = load_records(TRAIN_TASK1)
    t2_records = load_records(TRAIN_TASK2)
    clean_t1 = clean_task1_gold(t1_records, t2_records)
    clean_t2 = clean_task2_gold(t2_records)

    model_spec = SPECS[args.model]
    upstream_lineages: list[dict[str, Any]] = []

    # Validate every optional artifact before constructing an LM.
    direct = None
    if args.direct_task1 is not None:
        direct_lineage = load_artifact_lineage(
            args.direct_task1,
            expected_task=1,
            requested_track=args.track,
            expected_architecture=STAGE0_TASK1_ARCHITECTURE,
            expected_model=args.model,
            expected_output_kind="task-1-predictions",
            kind="direct-task1",
        )
        upstream_lineages.append(_lineage_without_manifest(direct_lineage))
        direct = load_task1_predictions(args.direct_task1, records)

    policy = None
    if args.fusion_policy is not None:
        policy_lineage = load_artifact_lineage(
            args.fusion_policy,
            expected_task=1,
            requested_track=args.track,
            expected_architecture=FUSION_ARCHITECTURE,
            expected_model=args.model,
            expected_output_kind="task1-fusion-policy",
            require_adopted=True,
            kind="task1-fusion-policy",
        )
        upstream_lineages.append(_lineage_without_manifest(policy_lineage))
        policy = _load_policy(args.fusion_policy)

    role_manifest: dict[str, Any] | None = None
    role_bundle: dict[str, Any] | None = None
    bundle_examples = None
    role_lineage = None
    if args.role_state is not None:
        role_lineage = load_artifact_lineage(
            args.role_state,
            expected_task=2,
            requested_track=args.track,
            expected_architecture=ARCHITECTURE_VERSION,
            expected_setting=args.setting,
            expected_model=args.model,
            expected_output_kind="v3-role-state",
            kind="compiled-role-state",
        )
        upstream_lineages.append(_lineage_without_manifest(role_lineage))
        bundle_lineage = load_artifact_lineage(
            args.role_state.parent / "role_bundle.json",
            expected_task=2,
            requested_track=args.track,
            expected_architecture=ARCHITECTURE_VERSION,
            expected_setting=args.setting,
            expected_model=args.model,
            expected_output_kind="v3-role-bundle",
            kind="compiled-role-bundle",
        )
        upstream_lineages.append(_lineage_without_manifest(bundle_lineage))
        role_manifest = role_lineage["manifest_data"]
        role_bundle, bundle_examples = load_role_bundle(args.role_state)
        for key, expected in {
            "model": args.model,
            "track": args.track,
            "setting": args.setting,
        }.items():
            if role_bundle.get(key) != expected:
                raise SystemExit(
                    f"role bundle {key}={role_bundle.get(key)!r}, requested {expected!r}"
                )
        if args.training_proposals is not None or args.memory_pool != "auto":
            raise SystemExit(
                "a role bundle owns its exact memory; do not pass --training-proposals "
                "or a non-auto --memory-pool"
            )

    bundle_settings = (role_bundle or {}).get("settings") or {}
    max_demos = _resolve_bundle_setting("max_demos", args.max_demos, bundle_settings, 7)
    context_chars = _resolve_bundle_setting("context_chars", args.context_chars, bundle_settings, 500)
    demo_context_chars = _resolve_bundle_setting(
        "demo_context_chars", args.demo_context_chars, bundle_settings, 350
    )
    granularity = _resolve_bundle_setting(
        "granularity", args.granularity, bundle_settings, "connective"
    )
    containment_threshold = _resolve_bundle_setting(
        "containment_threshold", args.containment_threshold, bundle_settings, 0.80
    )
    role_cot = _resolve_bundle_setting("role_cot", args.role_cot, bundle_settings, False)
    if max_demos < 0 or context_chars < 0 or demo_context_chars < 0:
        raise SystemExit("demo count and context sizes must be non-negative")
    if not 0.0 < containment_threshold <= 1.0:
        raise SystemExit("--containment-threshold must lie in (0, 1]")

    if role_bundle is not None:
        recorded_selector = "balanced" if bundle_settings.get("demo_selector") else "none"
        if args.demo_selector not in {"auto", recorded_selector}:
            raise SystemExit(
                f"--demo-selector={args.demo_selector!r} does not match role bundle "
                f"{recorded_selector!r}"
            )
        selector_mode = recorded_selector
        memory_examples = bundle_examples or []
        training_ids = list(role_manifest.get("training_ids", []))
        training_id_set = set(training_ids)
        if has_gold and training_id_set & inference_ids:
            overlap = sorted(training_id_set & inference_ids)
            raise SystemExit(
                "scored inference overlaps compiled training IDs; leakage would occur: "
                f"{overlap[:20]}"
            )
    else:
        selector_mode = "balanced" if args.demo_selector == "auto" else args.demo_selector
        train_ids, _ = train_val_ids(t1_records, t2_records)
        memory_pool = args.memory_pool
        if memory_pool == "auto":
            memory_pool = "all" if not has_gold else "legacy-train"
        candidate_memory_ids = sorted(clean_t1) if memory_pool == "all" else list(train_ids)
        # Remove the complete scored inference set, not merely the current query.
        if has_gold:
            candidate_memory_ids = [pid for pid in candidate_memory_ids if pid not in inference_ids]
        memory_records = [
            record
            for record in t1_records
            if record["paragraph_id"] in set(candidate_memory_ids)
            and (args.setting == "both" or record["type"] == args.setting)
        ]
        training_ids = [record["paragraph_id"] for record in memory_records]
        if selector_mode == "balanced" and not memory_records:
            raise SystemExit("balanced demo selection has no leakage-safe memory records")
        memory_proposals = None
        if selector_mode == "balanced":
            if args.training_proposals is None:
                raise SystemExit(
                    "uncompiled balanced v3 requires --training-proposals; "
                    "deterministic full-text atoms are a different architecture"
                )
            training_lineage = load_artifact_lineage(
                args.training_proposals,
                expected_task=2,
                requested_track=args.track,
                expected_architecture=STAGE0_QUOTE_ARCHITECTURE,
                expected_model=args.model,
                expected_output_kind="quote-proposals",
                kind="training-quote-proposals",
            )
            upstream_lineages.append(_lineage_without_manifest(training_lineage))
            memory_proposals = load_task2_predictions(args.training_proposals, memory_records)
        elif args.training_proposals is not None:
            raise SystemExit("--training-proposals is unused when --demo-selector=none")
        memory_examples = span_role_examples(
            memory_records,
            clean_t2,
            proposals=memory_proposals,
            granularity=granularity,
            containment_threshold=containment_threshold,
        ) if selector_mode == "balanced" else []

    proposals = None
    proposal_telemetry: dict[str, Any]
    if args.proposals is not None:
        proposal_lineage = load_artifact_lineage(
            args.proposals,
            expected_task=2,
            requested_track=args.track,
            expected_architecture=STAGE0_QUOTE_ARCHITECTURE,
            expected_model=args.model,
            expected_output_kind="quote-proposals",
            kind="inference-quote-proposals",
        )
        upstream_lineages.append(_lineage_without_manifest(proposal_lineage))
        proposals = load_task2_predictions(args.proposals, records)
        proposal_telemetry = {
            "source": str(args.proposals),
            "sha256": proposal_lineage["sha256"],
            "adapter": "ChatAdapter",
            "errors": [],
        }
    else:
        proposal_telemetry = {
            "source": "live",
            "adapter": "ChatAdapter",
            "cot": not args.no_extractor_cot,
            "errors": [],
        }

    optimizer_name = role_manifest.get("optimizer") if role_manifest else None
    reflection_spec = None
    prompt_spec = None
    if role_manifest:
        auxiliary = role_manifest.get("auxiliary_models") or {}
        if auxiliary.get("reflection"):
            reflection_spec = SPECS[auxiliary["reflection"]["key"]]
        if auxiliary.get("prompt"):
            prompt_spec = SPECS[auxiliary["prompt"]["key"]]
    sources = list((role_manifest or {}).get("data_sources", []))
    for source in ("daleel2026:train-task-1", "daleel2026:train-task-2"):
        if training_ids and source not in sources:
            sources.append(source)
    if source_id not in sources:
        sources.append(source_id)
    try:
        provenance = build_provenance_manifest(
            tasks=(1, 2),
            track=args.track,
            setting=args.setting,
            task_model=model_spec,
            optimizer=optimizer_name,
            training_ids=training_ids,
            data_sources=sources,
            architecture_version=ARCHITECTURE_VERSION,
            reflection_model=reflection_spec,
            prompt_model=prompt_spec,
        )
    except ComplianceError as exc:
        raise SystemExit(f"artifact/run provenance is not compliant: {exc}") from exc
    provenance["upstream_artifacts"] = upstream_lineages
    provenance["input"] = {
        "source_id": source_id,
        "path": str(source_path),
        "sha256": file_sha256(source_path),
        "selected_ids_sha256": training_ids_sha256(sorted(inference_ids)),
        "record_count": len(records),
    }
    provenance["data_artifacts"] = []
    if training_ids:
        provenance["data_artifacts"].extend(
            [
                {
                    "source_id": "daleel2026:train-task-1",
                    "path": str(TRAIN_TASK1),
                    "sha256": file_sha256(TRAIN_TASK1),
                },
                {
                    "source_id": "daleel2026:train-task-2",
                    "path": str(TRAIN_TASK2),
                    "sha256": file_sha256(TRAIN_TASK2),
                },
            ]
        )
    provenance["runtime_policy"] = {
        "max_extraction_error_rate": args.max_extraction_error_rate,
        "max_role_error_rate": args.max_role_error_rate,
        "live_proposals": args.live_proposals,
    }

    resolved_config = {
        "script": "run_decomposed.py",
        "argv": sys.argv[1:],
        "architecture": ARCHITECTURE_VERSION,
        "model": args.model,
        "track": args.track,
        "setting": args.setting,
        "source_id": source_id,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "selection": source_tag,
        "role_state": str(args.role_state) if args.role_state else None,
        "role_state_sha256": file_sha256(args.role_state) if args.role_state else None,
        "demo_selector": selector_mode,
        "memory_ids_sha256": training_ids_sha256(training_ids),
        "max_demos": max_demos,
        "context_chars": context_chars,
        "demo_context_chars": demo_context_chars,
        "granularity": granularity,
        "containment_threshold": containment_threshold,
        "role_cot": role_cot,
        "fusion": args.fusion,
        "fusion_policy": str(args.fusion_policy) if args.fusion_policy else None,
        "threads": args.threads,
        "max_tokens": args.max_tokens,
        "extractor_max_tokens": args.extractor_max_tokens,
        "extractor_cot": not args.no_extractor_cot,
        "extractor_adapter": "ChatAdapter",
        "role_adapter": "ChatAdapter",
        "max_extraction_error_rate": args.max_extraction_error_rate,
        "max_role_error_rate": args.max_role_error_rate,
        "dspy_version": dspy.__version__,
    }
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_ok": True,
                    "resolved_config": resolved_config,
                    "record_count": len(records),
                    "training_id_count": len(training_ids),
                    "upstream_artifacts": upstream_lineages,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"v3-decomposed-{args.model}-{args.setting}-{source_tag}",
        resolved_config,
    )
    config_path = atomic_write_json(exp_dir / "config.json", resolved_config)
    pending_provenance_path = atomic_write_json(
        exp_dir / "provenance.pending.json", provenance
    )

    # No provider object is constructed above this line.
    role_lm = make_lm(model_spec, max_tokens=args.max_tokens)
    if proposals is None:
        dspy.configure(
            lm=make_lm(model_spec, max_tokens=args.extractor_max_tokens),
            adapter=dspy.ChatAdapter(),
        )
        proposals, live_telemetry = run_quote_proposals(
            records,
            threads=args.threads,
            cot=not args.no_extractor_cot,
        )
        proposal_telemetry.update(live_telemetry)
        extraction_errors = len(live_telemetry["errors"])
        if _error_rate_exceeded(
            extraction_errors, len(records), args.max_extraction_error_rate
        ):
            failure_path = atomic_write_json(
                exp_dir / "FAILED.json",
                {
                    "stage": "quote-extraction",
                    "error_count": extraction_errors,
                    "record_count": len(records),
                    "allowed_rate": args.max_extraction_error_rate,
                    "telemetry": proposal_telemetry,
                },
            )
            raise SystemExit(f"quote extraction error-rate guard failed: {failure_path}")

    dspy.configure(lm=role_lm, adapter=dspy.ChatAdapter())
    selector = None
    if selector_mode == "balanced":
        selector = BalancedRoleDemoSelector(
            memory_examples,
            max_demos=max_demos,
            context_chars=demo_context_chars,
            same_genre_bonus=float(bundle_settings.get("same_genre_bonus", 0.05)),
        )
    role_program = SpanRoleDecision(
        cot=role_cot,
        demo_selector=selector,
        context_chars=context_chars,
    )
    if args.role_state is not None:
        role_program.load(args.role_state)

    inference_examples = span_role_examples(
        records,
        clean_t2 if has_gold else {},
        proposals=proposals,
        granularity=granularity,
        containment_threshold=containment_threshold,
    )
    started = time.time()
    if inference_examples:
        predictions, failed, exceptions = role_program.batch(
            inference_examples,
            num_threads=args.threads,
            max_errors=max(10, len(inference_examples) + 1),
            return_failed_examples=True,
            provide_traceback=False,
            timeout=0,
        )
    else:
        predictions, failed, exceptions = [], [], []
    role_seconds = time.time() - started
    decision_diagnostics = role_diagnostics(inference_examples, predictions)
    role_error_count = max(len(failed), decision_diagnostics["parse_failures"])
    if _error_rate_exceeded(
        role_error_count, len(inference_examples), args.max_role_error_rate
    ):
        failure_path = atomic_write_json(
            exp_dir / "FAILED.json",
            {
                "stage": "role-classification",
                "error_count": role_error_count,
                "atom_count": len(inference_examples),
                "allowed_rate": args.max_role_error_rate,
                "decision": decision_diagnostics,
                "failure_sample": [f"{type(exc).__name__}: {exc}" for exc in exceptions[:5]],
            },
        )
        raise SystemExit(f"role-classification error-rate guard failed: {failure_path}")

    span_predictions, span_label_predictions, assembly = assemble_example_predictions(
        inference_examples,
        predictions,
        fallback_to_draft=True,
    )
    for record in records:
        pid = record["paragraph_id"]
        span_predictions.setdefault(pid, [])
        span_label_predictions.setdefault(pid, set())

    final_labels = {
        record["paragraph_id"]: fuse_labels(
            span_label_predictions[record["paragraph_id"]],
            direct.get(record["paragraph_id"], set()) if direct is not None else None,
            args.fusion,
            policy,
        )
        for record in records
    }
    task1_rows = [
        {
            "paragraph_id": record["paragraph_id"],
            "text": record["text"],
            "type": record["type"],
            "labels": sorted(final_labels[record["paragraph_id"]]),
        }
        for record in records
    ]
    task2_rows = _proposal_rows(records, span_predictions)
    proposal_rows = _proposal_rows(records, proposals)
    problems = (
        validate_records_against_source(task1_rows, records, "task_1")
        + validate_records_against_source(task2_rows, records, "task_2")
        + validate_records_against_source(proposal_rows, records, "task_2")
    )
    if problems:
        raise SystemExit("source-bound output preflight failed:\n- " + "\n- ".join(problems[:30]))

    metrics: dict[str, Any] = {
        "architecture": ARCHITECTURE_VERSION,
        "model": args.model,
        "track": args.track,
        "setting": args.setting,
        "source_id": source_id,
        "n_paragraphs": len(records),
        "n_atoms": len(inference_examples),
        "role_seconds": round(role_seconds, 2),
        "demo_selector": selector_mode,
        "memory_decisions": len(memory_examples),
        "memory_paragraphs": len(set(training_ids)),
        "role_state": str(args.role_state) if args.role_state else None,
        "fusion": args.fusion,
        "fusion_policy": policy,
        "assembly": assembly,
        "decision": decision_diagnostics,
        "n_failed_role_calls": len(failed),
        "role_failure_sample": [f"{type(exc).__name__}: {exc}" for exc in exceptions[:5]],
        "proposals": proposal_telemetry,
    }
    if has_gold:
        ids = {record["paragraph_id"] for record in records}
        gold_t1 = {pid: clean_t1[pid] for pid in ids}
        gold_t2 = {pid: clean_t2[pid] for pid in ids}
        t1_score = task1_macro_f1(gold_t1, final_labels)
        t2_score = span_partial_f1(gold_t2, span_predictions)
        proposal_score = span_partial_f1(gold_t2, proposals)
        metrics.update(
            {
                "task1_macro_f1": t1_score["macro_f1"],
                "task1_micro_f1": t1_score["micro_f1"],
                "task1_per_label_f1": {
                    label: values["f1"] for label, values in t1_score["per_label"].items()
                },
                "task2_f1": t2_score["f1"],
                "task2_precision": t2_score["precision"],
                "task2_recall": t2_score["recall"],
                "task2_per_label_f1": {
                    label: values["f1"] for label, values in t2_score["per_label"].items()
                },
                "proposal_task2_f1": proposal_score["f1"],
                "task2_delta_vs_proposals": t2_score["f1"] - proposal_score["f1"],
            }
        )
        if direct is not None:
            metrics["direct_task1_macro_f1"] = task1_macro_f1(gold_t1, direct)["macro_f1"]

    prediction_dir = exp_dir / "predictions"
    task1_path = prediction_dir / "task_1.jsonl"
    task2_path = prediction_dir / "task_2.jsonl"
    proposals_path = prediction_dir / "quote_proposals.jsonl"
    write_jsonl(task1_path, task1_rows)
    write_jsonl(task2_path, task2_rows)
    write_jsonl(proposals_path, proposal_rows)
    metrics_path = atomic_write_json(exp_dir / "metrics.json", metrics)
    provenance["outputs"] = [
        output_record(exp_dir, task1_path, kind="v3-task1-predictions"),
        output_record(exp_dir, task2_path, kind="v3-task2-predictions"),
        output_record(exp_dir, proposals_path, kind="quote-proposals"),
    ]
    provenance_path = atomic_write_json(exp_dir / "provenance.json", provenance)
    pending_provenance_path.unlink(missing_ok=True)
    write_completion_marker(
        exp_dir,
        [config_path, metrics_path, provenance_path, task1_path, task2_path, proposals_path],
        metadata={"architecture": ARCHITECTURE_VERSION, "source_id": source_id},
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nTask 1 predictions: {task1_path}")
    print(f"Task 2 predictions: {task2_path}")


if __name__ == "__main__":
    main()
