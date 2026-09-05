#!/usr/bin/env python3
"""Validate a predictions jsonl and package it as a Codabench submission.

Usage (from shared-task/):
    uv run scripts/package_submission.py path/to/preds.jsonl \
        --team TeamName --setting editorial --task task_1

Writes submissions/<team>_<setting>.zip. Remember: for each setting, only
the LAST submission uploaded before the phase deadline is scored.
"""

import argparse
import sys
from pathlib import Path

from daleel.constants import TEAM_NAME, TRAINING_SETTINGS
from daleel.submission import TASK_FILENAMES, package_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="predictions file (jsonl)")
    parser.add_argument(
        "--team",
        default=TEAM_NAME,
        help=f"team name, no underscores (default: {TEAM_NAME})",
    )
    parser.add_argument("--setting", required=True, choices=TRAINING_SETTINGS)
    parser.add_argument("--task", default="task_1", choices=tuple(TASK_FILENAMES))
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="authoritative dev/test input JSONL; IDs, text, and type must match exactly",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "submissions",
        help="output directory (default: shared-task/submissions/)",
    )
    args = parser.parse_args()

    try:
        zip_path = package_submission(
            args.jsonl,
            args.team,
            args.setting,
            args.out,
            args.task,
            source_path=args.source,
        )
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"wrote {zip_path}")
    print("next: upload on Codabench and log the result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
