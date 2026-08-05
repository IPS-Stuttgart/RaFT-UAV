from __future__ import annotations

from pathlib import Path
import shlex

from raft_uav.calibration.nis_covariance import (
    ENV_NIS_COVARIANCE_CALIBRATION_JSON,
    environment_assignment,
)


def test_environment_assignment_quotes_shell_sensitive_paths() -> None:
    path = Path("outputs/calibration $(touch should-not-run) 'quoted'.json")

    assignment = environment_assignment(path)
    command = shlex.split(f"env {assignment} python -c pass")

    assert command[0] == "env"
    assert command[1] == f"{ENV_NIS_COVARIANCE_CALIBRATION_JSON}={path}"
    assert command[2:] == ["python", "-c", "pass"]
