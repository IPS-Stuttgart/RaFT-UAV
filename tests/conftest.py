"""Temporary pull-request diagnostics; removed before the final patch."""

from __future__ import annotations

import os
import subprocess
import sys


def pytest_sessionfinish(session, exitstatus) -> None:
    """Capture the full-suite failure in the repository's targeted-test artifact."""

    if os.environ.get("GITHUB_WORKFLOW") != "Debug MMUAD candidate mixture":
        return
    if os.environ.get("RAFT_UAV_FULL_SUITE_CHILD") == "1":
        return

    env = os.environ.copy()
    env["RAFT_UAV_FULL_SUITE_CHILD"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--maxfail=10",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    print("\n===== FULL SUITE DIAGNOSTIC =====")
    print(result.stdout)
    print(result.stderr)
    print("===== END FULL SUITE DIAGNOSTIC =====")
    if result.returncode != 0:
        session.exitstatus = result.returncode
