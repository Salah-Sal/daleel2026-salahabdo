"""Constrained relabel of Task 2 spans that contradict the routed Task 1 set
(HEADROOM_AUDIT.md P3).

The v2 blind relabel stage was flat (-0.002): re-deciding every span with
no new information cannot beat the original decision. This stage differs on
both counts: it touches ONLY spans whose label the tier-1 routed Task 1
system rejects for that paragraph (6-7% of spans), and it injects the
routed label set — externally validated at dev 0.6955, +0.05 stronger than
the span-derived sets — as a hard choice constraint (allowed labels or
NONE=drop).

Fold-0 free oracle bounds (2026-07-14, before any call): drop-violating
+0.018; relabel-to-best-allowed +0.040 (0.6644 -> 0.7042).

Pre-registered fold-0 screen: proceed to the remaining folds only if
relabeled F1 >= recorded + 0.015. Adoption for a dev probe follows the
standing +0.02 five-fold OOF gate against the adopted v3 baseline.

Example (from shared-task/):
  uv run scripts/t2_constrained_relabel.py --model gemma-4-31b-paid \
      --role-run experiments/<v3-fold0-gepa-run> \
      --v2-run experiments/<t1-routed-v2-gate-run> \
      --encoder-oof experiments/<quarter-oof-run>
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel import runtime as _runtime  # noqa: E402,F401 -- before dspy import

import dspy

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.candidates import marked_context
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.models import SPECS, make_lm
from daleel.policy import LABEL_POLICY, TASK_CONTEXT
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold

SCREEN_MARGIN = 0.015


class ChooseAllowedRole(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

A span extractor labeled the <TARGET> span with a type that a stronger,
independently validated paragraph-level classifier says is NOT present in
this paragraph. Re-decide the span's type, choosing ONLY among the allowed
types listed for this paragraph — or NONE if the span is not an
argumentative unit of any allowed type.

{LABEL_POLICY}"""

    paragraph_context: str = dspy.InputField(desc="paragraph with <TARGET> marks")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    target: str = dspy.InputField(desc="the exact span text")
    rejected_label: str = dspy.InputField(desc="the label the classifier rejects")
    allowed: str = dspy.InputField(desc="comma-separated allowed label codes")
    role: str = dspy.OutputField(
        desc="exactly one allowed code, or NONE to drop the span"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--role-run", type=Path, required=True,
                    help="v3 fold run: predictions/outer_task_2.jsonl is relabeled")
    ap.add_argument("--v2-run", type=Path, required=True,
                    help="route_task1_v2 gate run (composed OOF labels)")
    ap.add_argument("--encoder-oof", type=Path, required=True,
                    help="quarter OOF run (OT leg source for the v2.1 allowed sets)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    rec = {
        r["paragraph_id"]: [
            Span(i["start_offset"], i["end_offset"], i["label"]) for i in r["labels"]
        ]
        for r in read_jsonl(args.role_run / "predictions" / "outer_task_2.jsonl")
    }
    pids = set(rec)
    by_id = {
        r["paragraph_id"]: r
        for r in load_records(TRAIN_TASK1)
        if r["paragraph_id"] in pids
    }
    gold = {
        p: s
        for p, s in clean_task2_gold(load_records(TRAIN_TASK2)).items()
        if p in pids
    }

    # v2.1 allowed sets: the v2 composition with OT swapped to the encoder leg
    v2 = {
        r["paragraph_id"]: set(r["labels"])
        for r in read_jsonl(args.v2_run / "predictions" / "preds.jsonl")
    }
    enc = {
        r["paragraph_id"]: set(r["labels"])
        for r in read_jsonl(args.encoder_oof / "predictions" / "oof_task_1.jsonl")
    }
    allowed = {
        p: (v2[p] - {"OT"}) | ({"OT"} if "OT" in enc[p] else set()) for p in pids
    }

    recorded_f1 = span_partial_f1(gold, rec)["f1"]

    jobs = []  # (pid, span_index, span)
    for p in sorted(pids):
        for i, s in enumerate(rec[p]):
            if s.label not in allowed[p]:
                jobs.append((p, i, s))

    spec = SPECS[args.model]
    dspy.configure(lm=make_lm(spec, max_tokens=args.max_tokens),
                   adapter=dspy.ChatAdapter())
    judge = dspy.ChainOfThought(ChooseAllowedRole)
    valid_codes = {"CO", "AS", "TE", "ST", "AN", "OT"}

    def decide(job):
        pid, idx, span = job
        record = by_id[pid]
        allow = sorted(allowed[pid])
        last = None
        for attempt in range(4):
            try:
                pred = judge(
                    paragraph_context=marked_context(
                        record["text"], span.start, span.end, context_chars=500
                    ),
                    genre=record["type"],
                    target=record["text"][span.start:span.end],
                    rejected_label=span.label,
                    allowed=", ".join(allow) if allow else "NONE",
                )
                token = str(pred.role or "").strip().upper()
                if token in allow:
                    return pid, idx, token, None
                if token == "NONE" or token not in valid_codes:
                    return pid, idx, None, None  # drop
                # a valid code outside the allowed set: treat as drop
                return pid, idx, None, None
            except Exception as e:
                last = f"{pid}[{idx}]: {type(e).__name__}: {e}"
                time.sleep(min(75, 25 * (attempt + 1)))
        return pid, idx, span.label, last  # containment: keep unchanged

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(decide, jobs))
    wall = time.time() - t0
    errors = [e for *_, e in results if e]

    decision = {(pid, idx): label for pid, idx, label, _ in results}
    relabeled = {}
    n_reassigned = n_dropped = 0
    for p in pids:
        out = []
        for i, s in enumerate(rec[p]):
            if (p, i) not in decision:
                out.append(s)
                continue
            new = decision[(p, i)]
            if new is None:
                n_dropped += 1
            elif new == s.label:
                out.append(s)
            else:
                out.append(Span(s.start, s.end, new))
                n_reassigned += 1
        relabeled[p] = out

    official = span_partial_f1(gold, relabeled)
    screen_pass = official["f1"] >= recorded_f1 + SCREEN_MARGIN

    resolved_config = {
        "script": "t2_constrained_relabel.py",
        "argv": sys.argv[1:],
        "milestone": "HEADROOM_AUDIT.md P3",
        "model": spec.key,
        "litellm_id": spec.litellm_id,
        "role_run": str(args.role_run),
        "v2_run": str(args.v2_run),
        "encoder_oof": str(args.encoder_oof),
        "recorded_sha256": file_sha256(
            args.role_run / "predictions" / "outer_task_2.jsonl"
        ),
        "allowed_rule": "v2 composition with OT from encoder leg (v2.1)",
        "n_paragraphs": len(pids),
        "n_spans": sum(len(v) for v in rec.values()),
        "n_violating": len(jobs),
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"t2-crelabel-{spec.key}-n{len(pids)}",
        resolved_config,
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)
    metrics = {
        "recorded_f1": round(recorded_f1, 4),
        "relabeled_f1": round(official["f1"], 4),
        "delta": round(official["f1"] - recorded_f1, 4),
        "per_label_f1": {
            label: round(v["f1"], 4) for label, v in official["per_label"].items()
        },
        "n_violating": len(jobs),
        "n_reassigned": n_reassigned,
        "n_dropped": n_dropped,
        "n_kept_by_error": len(errors),
        "errors_sample": errors[:5],
        "wall_seconds": round(wall, 1),
        "screen": {
            "rule": f"relabeled >= recorded + {SCREEN_MARGIN}",
            "pass": screen_pass,
        },
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    write_jsonl(
        exp_dir / "predictions" / "relabeled_task_2.jsonl",
        [
            {
                "paragraph_id": p,
                "text": by_id[p]["text"],
                "type": by_id[p]["type"],
                "labels": [
                    {"label": s.label, "start_offset": s.start, "end_offset": s.end}
                    for s in relabeled[p]
                ],
            }
            for p in sorted(pids)
        ],
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nmetrics: {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
