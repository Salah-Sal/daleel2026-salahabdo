"""Test-session environment, applied before any daleel import.

MLflow autologging is ON by default for real runs (daleel.runtime), but the
test suite is offline and must stay side-effect-free: hard-disable it here
so no traces/experiments are written to mlruns/ during tests. conftest.py
loads before test modules, which is what guarantees the env var beats
daleel.runtime's import-time gate.
"""

import os

os.environ["DALEEL_MLFLOW"] = "0"
