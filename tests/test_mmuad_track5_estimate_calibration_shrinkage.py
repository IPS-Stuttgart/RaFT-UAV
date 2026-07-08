from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_calibration import fit_track5_estimate_calibration
from raft_uav.mmuad.track5_estimate_calibration_shrinkage import (
    apply_track5_estimate_calibration_shrinkage,
    search_track5_estimate_calibration_alpha,
    write_track5_estimate_calibration_shrinkage_outputs,
)
from raft_uav.mmuad.track5_estimate_calibration_shrinkage_cli import main as shrinkage_main


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "x_m": [0.0, 1.0, 4.0],
            "y_m": [0.0, 0.0, 4.0],
            "z_m": [0.0, 0.0, 4.0],
        }
    )


def _biased_estimates() -> pd.DataFrame:
    truth = _truth()
    return pd.DataFrame(
        {
            "sequence_id": truth["sequence_id"],
            "time_s": truth["time_s"],
            "state_x_m": truth["x_m"] + 2.0,
            "state_y_m": truth["y_m"] - 4.0,
            "state_z_m": truth["z_m"] + 6.0,
        }
    )


def _calibration() -> dict:
    config, _ = fit_track5_estimate_calibration(
        _biased_estimates(),
        template=_template(),
        truth=_truth(),
        mode="translation",
    )
    return config


def test_apply_calibration_shrinkage_blends_raw_and_full_correction() -> None:
    shrunk, diagnostics = apply_track5_estimate_calibration_shrinkage(
        _biased_estimates(),
        template=_template(),
        calibration=_calibration(),
        alpha=0.5,
    )

    assert shrunk["state_x_m"].tolist() == pytest.approx([1.0, 2.0, 5.0])
    assert shrunk["state_y_m"].tolist() == pytest.approx([-2.0, -2.0, 2.0])
    assert shrunk["state_z_m"].tolist() == pytest.approx([3.0, 3.0, 7.0])
    assert shrunk["track5_estimate_calibration_alpha"].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert diagnostics["shrunk_row_valid"].all()


def test_calibration_shrinkage_search_selects_full_correction_when_best() -> None:
    grid, best = search_track5_estimate_calibration_alpha(
        _biased_estimates(),
        template=_template(),
        truth=_truth(),
        calibration=_calibration(),
        alpha_values=(0.0, 0.5, 1.0),
    )

    assert grid["alpha"].tolist() == [0.0, 0.5, 1.0]
    assert best["alpha"] == pytest.approx(1.0)
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)


def test_calibration_shrinkage_outputs_leaderboard_ready_zip(tmp_path: Path) -> None:
    paths = write_track5_estimate_calibration_shrinkage_outputs(
        estimates=_biased_estimates(),
        template=_template(),
        calibration=_calibration(),
        output_dir=tmp_path / "apply",
        alpha=1.0,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["valid_shrunk_rows"] == 3
    assert manifest["alpha"] == pytest.approx(1.0)
    assert paths["official_zip"].exists()


def test_calibration_shrinkage_cli_search_and_apply_best_alpha(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    calibration_json = tmp_path / "calibration.json"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _biased_estimates().to_csv(estimates_csv, index=False)
    _template().to_csv(template_csv, index=False)
    _truth().to_csv(truth_csv, index=False)
    calibration_json.write_text(json.dumps(_calibration()), encoding="utf-8")
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = shrinkage_main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--calibration-json",
            str(calibration_json),
            "--truth-csv",
            str(truth_csv),
            "--alpha-grid",
            "0,0.5,1",
            "--use-best-alpha",
            "--write-apply",
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "search" / "mmuad_track5_calibration_shrinkage_grid.csv").exists()
    assert (output_dir / "search" / "mmuad_track5_calibration_shrinkage_best_alpha.json").exists()
    assert (output_dir / "apply" / "ug2_submission.zip").exists()


def test_calibration_shrinkage_cli_preserves_numeric_looking_sequence_ids(
    tmp_path: Path,
) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    calibration_json = tmp_path / "calibration.json"
    output_dir = tmp_path / "out"
    estimates_csv.write_text(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n"
        "001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    ).to_csv(template_csv, index=False)
    calibration_json.write_text(json.dumps(_calibration()), encoding="utf-8")

    status = shrinkage_main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--calibration-json",
            str(calibration_json),
            "--output-dir",
            str(output_dir),
            "--alpha",
            "0.0",
            "--default-classification",
            "2",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    rows = pd.read_csv(
        output_dir / "apply" / "mmuad_track5_calibration_shrunk_estimates.csv",
        dtype=str,
        keep_default_na=False,
    )
    validation = json.loads(
        (output_dir / "apply" / "mmuad_track5_calibration_shrinkage_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert rows.loc[0, "sequence_id"] == "001"
    assert float(rows.loc[0, "state_x_m"]) == pytest.approx(1.0)
    assert validation["leaderboard_ready"] is True


def test_calibration_shrinkage_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-calibration-shrinkage"]
        == "raft_uav.mmuad.track5_estimate_calibration_shrinkage_cli:main"
    )
