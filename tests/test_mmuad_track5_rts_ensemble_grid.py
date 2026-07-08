from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd

from raft_uav.mmuad.track5_rts_ensemble_grid import main as rts_grid_main
from raft_uav.mmuad.track5_rts_ensemble_grid import run_track5_rts_ensemble_grid_search
from raft_uav.mmuad.track5_rts_ensemble_grid import write_track5_rts_ensemble_grid_outputs
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 5,
            "Timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2] * 5,
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y_m": [0.0] * 5,
            "z_m": [1.0] * 5,
        }
    )


def _estimate(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0 + offset, 1.0 + offset, 8.0 + offset, 3.0 + offset, 4.0 + offset],
            "state_y_m": [0.0] * 5,
            "state_z_m": [1.0] * 5,
        }
    )


def test_rts_ensemble_grid_search_ranks_parameter_rows(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate().to_csv(estimate_csv, index=False)

    grid, best = run_track5_rts_ensemble_grid_search(
        [parse_estimate_spec(f"base={estimate_csv}")],
        template=_template(),
        truth=_truth(),
        measurement_sigma_grid=(5.0, 10.0),
        process_accel_grid=(0.5, 5.0),
        spread_variance_scale_grid=(0.0,),
        score_time_tolerance_s=1.0e-9,
    )

    assert len(grid) == 4
    assert grid.iloc[0]["pose_mse_m2"] <= grid.iloc[-1]["pose_mse_m2"]
    assert best["best"]["matched_row_count"] == 5
    assert best["best"]["measurement_sigma_m"] in {5.0, 10.0}


def test_rts_ensemble_grid_writes_best_submission(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    class_map_csv = tmp_path / "class_map.csv"
    _estimate().to_csv(estimate_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )
    paths = write_track5_rts_ensemble_grid_outputs(
        estimate_inputs=[parse_estimate_spec(f"base={estimate_csv}")],
        template=_template(),
        truth=_truth(),
        output_dir=tmp_path / "out",
        measurement_sigma_grid=(5.0,),
        process_accel_grid=(1.0,),
        spread_variance_scale_grid=(0.0,),
        write_best_submission=True,
        class_map={"seq0001": "2"},
    )

    assert paths["grid_csv"].exists()
    assert paths["best_json"].exists()
    assert paths["best_official_zip"].exists()
    with ZipFile(paths["best_official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    best = json.loads(paths["best_json"].read_text(encoding="utf-8"))["best"]
    assert best["matched_row_count"] == 5


def test_rts_ensemble_grid_cli_writes_outputs(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _estimate().to_csv(estimate_csv, index=False)
    _template().to_csv(template_csv, index=False)
    _truth().to_csv(truth_csv, index=False)

    status = rts_grid_main(
        [
            "--estimate-csv",
            f"base={estimate_csv}",
            "--template",
            str(template_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--measurement-sigma-grid",
            "5",
            "--process-accel-grid",
            "1,3",
            "--spread-variance-scale-grid",
            "0",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_rts_ensemble_grid.csv").exists()
    assert (output_dir / "mmuad_track5_rts_ensemble_grid_best.json").exists()


def test_rts_ensemble_grid_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-rts-ensemble-grid"]
        == "raft_uav.mmuad.track5_rts_ensemble_grid:main"
    )
