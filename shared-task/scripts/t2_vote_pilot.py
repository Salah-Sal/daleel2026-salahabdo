"""Task 2 atom-vote self-consistency pilot (HEADROOM_AUDIT.md P2).

The v2/v3 error anatomy says every Task 2 error is a wrong label on
correctly-extracted text, and identical T=0 champion runs swing +-0.03-0.04
macro — label noise that per-atom majority voting can average away. The
rollout infrastructure exists for Task 2 but no aggregator does; this pilot
builds one for a SINGLE outer fold as the pre-registered spend gate for the
five-fold campaign.

Design (frozen before the first replica ran):
- inputs: one completed v3 fold compile run (its bundle-bound role state)
  plus the same frozen proposals it consumed; inference over that fold's
  outer paragraphs only — the state never trained on them;
- k replicas of the role stage at temperature 0.7, rollout ids base..base+k-1
  (cache-busting; each replica itself cached and reproducible);
- per-atom vote: label fires iff >= floor(k/2)+1 parsed replicas emit it;
  an atom with zero parsed replicas falls back to extractor drafts exactly
  like a single-run parse failure;
- outcomes reported: recorded single-run outer F1 (from the fold run),
  each replica's own F1 (the variance being harvested), the majority-vote
  F1, and a plurality diagnostic (top label per atom, replica-0 tiebreak).

Spend gate for the full campaign (not an adoption rule): voted F1 >=
recorded outer F1 + 0.005 AND voted F1 >= mean(replica F1s).

Example (from shared-task/):
  uv run scripts/t2_vote_pilot.py --model gemma-4-31b-paid \
      --role-run experiments/<v3-fold0-gepa-run> \
      --proposals experiments/<t2-alltrain-run>/predictions/preds.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel import runtime as _runtime  # noqa: E402,F401 -- before dspy import

import dspy

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.candidates import assemble_role_spans, atomize_spans
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.dspy_span_roles import (
    BalancedRoleDemoSelector,
    SpanRoleDecision,
    parse_role_set,
    span_role_examples,
)
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.models import SPECS, make_lm
from daleel.role_artifacts import load_role_bundle
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold


def _load_proposals(path: Path, wanted: set[int]) -> dict[int, list[Span]]:
    return {
        row["paragraph_id"]: [
            Span(item["start_offset"], item["end_offset"], item["label"])
            for item in row["labels"]
        ]
        for row in read_jsonl(path)
        if row["paragraph_id"] in wanted
    }


def _spans_from_rows(path: Path) -> dict[int, list[Span]]:
    return {
        row["paragraph_id"]: [
            Span(item["start_offset"], item["end_offset"], item["label"])
            for item in row["labels"]
        ]
        for row in read_jsonl(path)
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--role-run", type=Path, required=True,
                    help="completed v3 fold compile run (compiled/role.json inside)")
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--rollout-base", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--limit", type=int, default=None, help="first N paragraphs (smoke)")
    args = ap.parse_args()
    if args.replicas < 3 or args.replicas % 2 == 0:
        raise SystemExit("--replicas must be an odd number >= 3")

    role_state = args.role_run / "compiled" / "role.json"
    bundle, memory_examples = load_role_bundle(role_state)
    settings = bundle.get("settings") or {}
    granularity = settings.get("granularity", "connective")
    containment = settings.get("containment_threshold", 0.80)

    outer_rows = read_jsonl(args.role_run / "predictions" / "outer_task_1.jsonl")
    outer_pids = [r["paragraph_id"] for r in outer_rows]
    if args.limit:
        outer_pids = outer_pids[: args.limit]
    wanted = set(outer_pids)

    t1_records = load_records(TRAIN_TASK1)
    records = [r for r in t1_records if r["paragraph_id"] in wanted]
    gold_t2 = {
        pid: spans
        for pid, spans in clean_task2_gold(load_records(TRAIN_TASK2)).items()
        if pid in wanted
    }
    proposals = _load_proposals(args.proposals, wanted)

    recorded = _spans_from_rows(args.role_run / "predictions" / "outer_task_2.jsonl")
    recorded = {pid: spans for pid, spans in recorded.items() if pid in wanted}
    recorded_f1 = span_partial_f1(gold_t2, recorded)["f1"]

    selector = None
    if settings.get("demo_selector") and memory_examples is not None:
        selector = BalancedRoleDemoSelector(
            memory_examples,
            max_demos=settings.get("max_demos", 7),
            context_chars=settings.get("demo_context_chars", 350),
            same_genre_bonus=float(settings.get("same_genre_bonus", 0.05)),
        )
    role_program = SpanRoleDecision(
        cot=settings.get("role_cot", False),
        demo_selector=selector,
        context_chars=settings.get("context_chars", 500),
    )
    role_program.load(role_state)

    examples = span_role_examples(
        records,
        gold_t2,
        proposals=proposals,
        granularity=granularity,
        containment_threshold=containment,
    )

    spec = SPECS[args.model]
    replica_parsed: list[list[tuple[tuple[str, ...], bool]]] = []
    replica_f1: list[float] = []
    wall: list[float] = []
    for r in range(args.replicas):
        lm = make_lm(
            spec,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            rollout_id=args.rollout_base + r,
        )
        dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
        t0 = time.time()
        predictions, failed, _ = role_program.batch(
            examples,
            num_threads=args.threads,
            max_errors=max(10, len(examples) + 1),
            return_failed_examples=True,
            provide_traceback=False,
            timeout=0,
        )
        wall.append(round(time.time() - t0, 1))
        parsed = []
        for pred in predictions:
            if pred is None:
                parsed.append(((), False))
            else:
                roles, ok, _unknown = parse_role_set(pred)
                parsed.append((roles, ok))
        replica_parsed.append(parsed)

        # per-replica assembly + score
        by_pid: dict[int, list[tuple[tuple[str, ...], bool]]] = defaultdict(list)
        for example, decision in zip(examples, parsed):
            by_pid[example.paragraph_id].append(decision)
        pred_spans: dict[int, list[Span]] = {}
        for record in records:
            pid = record["paragraph_id"]
            atoms = atomize_spans(record["text"], proposals.get(pid, ()), granularity)
            spans, _stats = assemble_role_spans(atoms, by_pid.get(pid, []))
            pred_spans[pid] = spans
        replica_f1.append(round(span_partial_f1(gold_t2, pred_spans)["f1"], 4))
        print(f"replica {r}: f1={replica_f1[-1]} wall={wall[-1]}s "
              f"failed={len(failed)}", file=sys.stderr)

    majority = args.replicas // 2 + 1

    def vote_decisions(index: int) -> tuple[tuple[str, ...], bool]:
        counts: Counter[str] = Counter()
        n_parsed = 0
        for parsed in replica_parsed:
            roles, ok = parsed[index]
            if not ok:
                continue
            n_parsed += 1
            counts.update(set(roles))
        if n_parsed == 0:
            return (), False  # falls back to drafts like a single-run failure
        fired = tuple(sorted(label for label, c in counts.items() if c >= majority))
        return fired, True

    def plurality_decisions(index: int) -> tuple[tuple[str, ...], bool]:
        counts: Counter[str] = Counter()
        n_parsed = 0
        for parsed in replica_parsed:
            roles, ok = parsed[index]
            if not ok:
                continue
            n_parsed += 1
            counts.update(set(roles))
        if n_parsed == 0:
            return (), False
        if not counts:
            return (), True
        top = max(counts.values())
        if top >= majority:
            return tuple(sorted(l for l, c in counts.items() if c >= majority)), True
        # no majority: replica-0 anchor
        return replica_parsed[0][index]

    scores = {}
    voted_spans_out: dict[int, list[Span]] = {}
    for name, decide in (("majority_vote", vote_decisions),
                         ("plurality_anchor0", plurality_decisions)):
        flat = [decide(i) for i in range(len(examples))]
        by_pid = defaultdict(list)
        for example, decision in zip(examples, flat):
            by_pid[example.paragraph_id].append(decision)
        pred_spans = {}
        for record in records:
            pid = record["paragraph_id"]
            atoms = atomize_spans(record["text"], proposals.get(pid, ()), granularity)
            spans, _stats = assemble_role_spans(atoms, by_pid.get(pid, []))
            pred_spans[pid] = spans
        official = span_partial_f1(gold_t2, pred_spans)
        scores[name] = {
            "f1": round(official["f1"], 4),
            "per_label_f1": {
                label: round(v["f1"], 4) for label, v in official["per_label"].items()
            },
        }
        if name == "majority_vote":
            voted_spans_out = pred_spans

    mean_replica = statistics.mean(replica_f1)
    gate = {
        "rule": "voted >= recorded_outer + 0.005 AND voted >= mean(replica f1)",
        "recorded_outer_f1": round(recorded_f1, 4),
        "mean_replica_f1": round(mean_replica, 4),
        "replica_f1_sd": round(statistics.pstdev(replica_f1), 4),
        "voted_f1": scores["majority_vote"]["f1"],
        "pass": (
            scores["majority_vote"]["f1"] >= recorded_f1 + 0.005
            and scores["majority_vote"]["f1"] >= mean_replica
        ),
    }

    resolved_config = {
        "script": "t2_vote_pilot.py",
        "argv": sys.argv[1:],
        "milestone": "HEADROOM_AUDIT.md P2",
        "model": spec.key,
        "litellm_id": spec.litellm_id,
        "role_run": str(args.role_run),
        "role_state_sha256": file_sha256(role_state),
        "proposals_sha256": file_sha256(args.proposals),
        "replicas": args.replicas,
        "rollout_base": args.rollout_base,
        "temperature": args.temperature,
        "granularity": granularity,
        "containment_threshold": containment,
        "n_outer_paragraphs": len(records),
        "n_atoms": len(examples),
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"t2-vote-pilot-k{args.replicas}-n{len(records)}",
        resolved_config,
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)
    metrics = {
        "recorded_outer_f1": round(recorded_f1, 4),
        "replica_f1": replica_f1,
        "replica_wall_seconds": wall,
        "scores": scores,
        "spend_gate": gate,
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    write_jsonl(
        exp_dir / "predictions" / "voted_task_2.jsonl",
        [
            {
                "paragraph_id": record["paragraph_id"],
                "text": record["text"],
                "type": record["type"],
                "labels": [
                    {"label": s.label, "start_offset": s.start, "end_offset": s.end}
                    for s in voted_spans_out.get(record["paragraph_id"], [])
                ],
            }
            for record in records
        ],
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nmetrics: {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
