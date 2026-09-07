"""Regression tests for Track 5 temporal-repair output preflight."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad import track5_temporal_repair as temporal_repair


def test_output_api_requires_template_before_any_writes(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="require_leaderboard_ready.*requires.*template"):
        temporal_repair.write_track5_temporal_repair_outputs(
            repaired=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            output_dir=output_dir,
            input_submission_path=tmp_path / "input.csv",
            require_leaderboard_ready=True,
        )

    assert not output_dir.exists()


def test_cli_requires_template_before_creating_outputs(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0],
            "Position": ["(0, 0, 0)", "(1, 0, 0)", "(2, 0, 0)"],
            "Classification": [2, 2, 2],
        }
    ).to_csv(submission_path, index=False)

    with pytest.raises(ValueError, match="require_leaderboard_ready.*requires.*template"):
        temporal_repair.main(
            [
                "--submission",
                str(submission_path),
                "--output-dir",
                str(output_dir),
                "--require-leaderboard-ready",
            ]
        )

    assert not output_dir.exists()


def test_missing_template_preflight_does_not_overwrite_existing_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sentinel = output_dir / temporal_repair.MANIFEST_JSON
    sentinel.write_bytes(b"previous successful output\n")

    with pytest.raises(ValueError, match="require_leaderboard_ready.*requires.*template"):
        temporal_repair.write_track5_temporal_repair_outputs(
            repaired=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            output_dir=output_dir,
            input_submission_path=tmp_path / "input.csv",
            require_leaderboard_ready=True,
        )

    assert sentinel.read_bytes() == b"previous successful output\n"
    assert list(output_dir.iterdir()) == [sentinel]
