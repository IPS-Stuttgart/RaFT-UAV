from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_temporal_repair_search import main as repair_search_main
from raft_uav.mmuad.track5_temporal_repair_search import search_track5_temporal_repair_parameters
from raft_uav.mmuad.track5_temporal_repair_search import write_temporal_repair_search_outputs


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0],
            "Position": ["(0, 0, 0)", "(100, 0, 0)", "(2, 0, 0)"],
            "Classification": [2, 2, 2],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        }
    )


def test_temporal_repair_search_selects_spike_repair_setting() -> None:
    grid, best = search_track5_temporal_repair_parameters(
        _submission_rows(),
        _truth_rows(),
        max_speed_grid=(50.0, 200.0),
        interpolation_residual_grid=(5.0,),
        iterations_grid=(1,),
    )

    assert len(grid) == 2
    assert best["max_speed_mps"] == pytest.approx(50.0)
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)
    no_repair = grid.loc[grid["max_speed_mps"] == 200.0].iloc[0]
    assert no_repair["repaired_row_count"] == 0
    assert no_repair["pose_mse_m2"] > 1000.0


def test_temporal_repair_search_materializes_one_shot_grid_iterables() -> None:
    max_speeds = (value for value in (50.0, 200.0))
    interpolation_residuals = (value for value in (5.0, 10.0))
    iterations = (value for value in (1, 2))

    grid, _ = search_track5_temporal_repair_parameters(
        _submission_rows(),
        _truth_rows(),
        max_speed_grid=max_speeds,
        interpolation_residual_grid=interpolation_residuals,
        iterations_grid=iterations,
    )

    actual = set(
        grid[["max_speed_mps", "max_interpolation_residual_m", "iterations"]]
        .itertuples(index=False, name=None)
    )
    expected = set(product((50.0, 200.0), (5.0, 10.0), (1, 2)))
    assert actual == expected


def test_temporal_repair_search_writes_best_artifacts(tmp_path: Path) -> None:
    paths = write_temporal_repair_search_outputs(
        submission=_submission_rows(),
        truth=_truth_rows(),
        output_dir=tmp_path,
        input_submission_path=tmp_path / "input.csv",
        max_speed_grid=(50.0, 200.0),
        interpolation_residual_grid=(5.0,),
        iterations_grid=(1,),
        write_best_submission=True,
        template=_submission_rows(),
        require_leaderboard_ready=True,
    )

    assert paths["grid_csv"].exists()
    assert paths["best_config_json"].exists()
    assert paths["best_zip"].exists()
    best = json.loads(paths["best_config_json"].read_text(encoding="utf-8"))
    assert best["max_speed_mps"] == 50.0
    assert best["metrics"]["pose_mse_m2"] == 0.0


def test_temporal_repair_search_cli_writes_grid(tmp_path: Path) -> None:
    submission_csv = tmp_path / "submission.csv"
    truth_csv = tmp_path / "truth.csv"
    out = tmp_path / "out"
    _submission_rows().to_csv(submission_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = repair_search_main(
        [
            "--submission",
            str(submission_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(out),
            "--max-speed-grid",
            "50,200",
            "--interpolation-residual-grid",
            "5",
            "--iterations-grid",
            "1",
        ]
    )

    assert status == 0
    assert (out / "mmuad_track5_temporal_repair_search_grid.csv").exists()
    assert (out / "mmuad_track5_temporal_repair_best_config.json").exists()


def test_temporal_repair_search_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-temporal-repair-search"]
        == "raft_uav.mmuad.track5_temporal_repair_search:main"
    )
