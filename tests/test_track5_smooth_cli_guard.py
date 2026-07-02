import pytest

from raft_uav.mmuad.track5_trajectory_smooth import main


def test_readiness_flag_requires_template(tmp_path):
    args = [
        "--submission",
        str(tmp_path / "input.csv"),
        "--output-dir",
        str(tmp_path / "out"),
        "--require-leaderboard-ready",
    ]
    with pytest.raises(SystemExit, match="template"):
        main(args)
