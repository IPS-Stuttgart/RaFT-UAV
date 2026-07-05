from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.track5_submission_ensemble import main as ensemble_main


def test_track5_submission_ensemble_require_leaderboard_ready_needs_template(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        ensemble_main(
            [
                "--submission",
                "a=1:/does/not/matter.csv",
                "--output-dir",
                str(tmp_path / "out"),
                "--require-leaderboard-ready",
            ]
        )

    assert excinfo.value.code == 2
