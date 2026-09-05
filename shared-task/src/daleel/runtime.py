"""Process setup that must happen BEFORE `import dspy`.

Import this module first in anything that touches DSPy (the daleel.dspy_*
modules do so themselves):

- DSPY_CACHEDIR is pointed into `shared-task/experiments/` (gitignored) so
  compile/eval reruns hit the disk cache and interrupted runs resume free
  (D13/D15) — DSPy reads the variable at import time, hence this module.
- API keys are loaded from the repo-root `.env` (OPENROUTER_API_KEY et
  al.); values already present in the environment win.
- MLflow DSPy autologging is enabled BY DEFAULT into a local gitignored
  SQLite store, `shared-task/mlruns/mlflow.db` (mlflow >= 3.14 has the
  old file store in maintenance mode, so SQLite is the supported backend).
  Browse with `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
  Set DALEEL_MLFLOW=0 to disable — the test suite does (tests/conftest.py)
  so no traces are written offline. Must run AFTER the cache-dir env var
  above: autolog imports dspy.
"""

import os
from pathlib import Path

SHARED_TASK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SHARED_TASK_ROOT.parent
EXPERIMENTS_DIR = SHARED_TASK_ROOT / "experiments"

os.environ.setdefault("DSPY_CACHEDIR", str(EXPERIMENTS_DIR / ".dspy_cache"))


MLRUNS_DIR = SHARED_TASK_ROOT / "mlruns"


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines; '#' comments; no quoting)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def enable_mlflow() -> bool:
    """Turn on MLflow's native DSPy tracing (module/LM/adapter call trees).

    Default ON; DALEEL_MLFLOW=0/false/off disables (the gate sits before
    the import so disabled runs never pay mlflow's import cost). Contained:
    a missing or broken mlflow must never block a run — tracing is
    observability, not a dependency of correctness.
    """
    if os.environ.get("DALEEL_MLFLOW", "1").strip().lower() in ("0", "false", "off"):
        return False
    try:
        import mlflow
        import mlflow.dspy

        experiment = "daleel2026-shared-task"
        MLRUNS_DIR.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
        if mlflow.get_experiment_by_name(experiment) is None:
            # Pin the artifact root: the SQLite store's default is
            # CWD-relative, and scripts run from varying directories.
            mlflow.create_experiment(
                experiment, artifact_location=(MLRUNS_DIR / "artifacts").as_uri()
            )
        mlflow.set_experiment(experiment)
        mlflow.dspy.autolog()
        return True
    except Exception:
        return False


load_env()
MLFLOW_ENABLED = enable_mlflow()
