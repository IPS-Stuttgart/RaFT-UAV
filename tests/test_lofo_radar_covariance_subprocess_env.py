from __future__ import annotations

from pathlib import Path


def test_lofo_covariance_runner_uses_shared_subprocess_env_helper() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_lofo_radar_covariance_tuning.py"
    )
    source = script.read_text(encoding="utf-8")

    assert "common.subprocess_env()" in source
    assert "os.environ.copy()" not in source
