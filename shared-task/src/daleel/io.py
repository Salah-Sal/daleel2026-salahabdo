"""JSONL reading/writing with the encoding conventions this task needs.

Arabic text must round-trip readably: write with ensure_ascii=False so the
files stay human-inspectable, and always use UTF-8 regardless of locale.
"""

import json
import os
from pathlib import Path
import tempfile


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Atomically replace ``path`` with UTF-8 JSON Lines records.

    Model runs are interruptible and their outputs contain expensive results.
    Writing through a temporary file in the destination directory prevents a
    killed process from leaving a valid-looking, partially written JSONL file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
