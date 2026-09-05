"""Tabulate the D10 bake-off from experiments/*/metrics.json.

Collects every val run produced by run_zero_shot.py, joins Task 1 and
Task 2 rows per model, and prints a markdown table sorted by Task 1
official macro-F1. Gate/score columns follow the pre-registered D10
criterion: G1 = format compliance (fail < 0.98), S1 = official score,
S2 = zero-shot ST/CO recall.

Usage (from shared-task/): uv run scripts/bakeoff_table.py [--sort t2]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.models import SPECS
from daleel.runtime import EXPERIMENTS_DIR


def collect() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(EXPERIMENTS_DIR.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if m.get("on") != "val" or m.get("program", "seed") != "seed" or m.get("compiled"):
            continue  # quote-program/compiled runs are reported in their own notes
        key = m["model"]
        row = rows.setdefault(key, {"model": key})
        spec = SPECS.get(key)
        row["params_b"] = spec.params_b if spec else None
        row["closed"] = spec.closed_track if spec else None
        if m["task"] == 1:
            row["t1"] = m.get("s1_official_macro_f1")
            row["t1_g1"] = m.get("g1_format_compliance")
            rare = m.get("s2_rare_recall", {})
            row["st_recall"], row["co_recall"] = rare.get("ST"), rare.get("CO")
        else:
            row["t2"] = m.get("s1_official_span_f1")
            row["t2_g1"] = m.get("g1_format_compliance")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", choices=("t1", "t2", "size"), default="t1")
    args = ap.parse_args()
    rows = list(collect().values())
    sort_key = {
        "t1": lambda r: -(r.get("t1") or 0),
        "t2": lambda r: -(r.get("t2") or 0),
        "size": lambda r: r.get("params_b") or 0,
    }[args.sort]
    rows.sort(key=sort_key)

    def fmt(x, pct=False):
        if x is None:
            return "—"
        return f"{x:.3f}" if not pct else f"{x:.2f}"

    print("| model | B | track | T1 macro-F1 | T1 G1 | ST rec | CO rec | T2 span-F1 | T2 G1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        track = {True: "closed", False: "open", None: "?"}[r.get("closed")]
        print(
            f"| {r['model']} | {r.get('params_b') or '?'} | {track} "
            f"| {fmt(r.get('t1'))} | {fmt(r.get('t1_g1'), pct=True)} "
            f"| {fmt(r.get('st_recall'), pct=True)} | {fmt(r.get('co_recall'), pct=True)} "
            f"| {fmt(r.get('t2'))} | {fmt(r.get('t2_g1'), pct=True)} |"
        )


if __name__ == "__main__":
    main()
