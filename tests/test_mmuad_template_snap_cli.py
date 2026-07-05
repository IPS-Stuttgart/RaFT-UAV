from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_template_snap_cli_module_execution_shows_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-m", "raft_uav.mmuad.template_snap_cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "official CSV/ZIP to snap" in result.stdout
    assert "--require-leaderboard-ready" in result.stdout
