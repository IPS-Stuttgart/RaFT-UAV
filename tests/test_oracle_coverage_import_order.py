from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_radar_association_import_does_not_reenter_oracle_coverage() -> None:
    """Baseline startup must finish before oracle-coverage patches are installed."""

    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_root = str(repository_root / "src")
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not inherited_pythonpath
        else os.pathsep.join((source_root, inherited_pythonpath))
    )
    command = "\n".join(
        (
            "import raft_uav.baselines.radar_association",
            "import raft_uav.evaluation.oracle_coverage as coverage",
            "assert getattr(coverage, '_sequence_scope_patch_applied', False)",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
