from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_calibration import (
    _fit_pairs,
    apply_track5_estimate_calibration,
    fit_track5_estimate_calibration,
    main as calibration_main,
    write_track5_estimate_calibration_apply_outputs,
)


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
            "state_x_m": truth["x_m"] + 1.5,
            "state_y_m": truth["y_m"] - 2.0,
            "state_z_m": truth["z_m"] + 0.5,
            "classification": [2, 2, 1],
        }
    )


def test_fit_pairs_rejects_nonfinite_timestamp_matches() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, float("nan")],
            "state_x_m": [1.5, 9.0],
            "state_y_m": [-2.0, 9.0],
            "state_z_m": [0.5, 9.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, float("nan")],
            "x_m": [0.0, -9.0],
            "y_m": [0.0, -9.0],
            "z_m": [0.0, -9.0],
        }
    )

    pairs = _fit_pairs(estimates, truth)

    assert len(pairs) == 1
    assert pairs["time_s"].tolist() == [0.0]
    assert pairs["truth_x_m"].tolist() == [0.0]


def test_fit_translation_calibration_removes_global_bias() -> None:
    config, pairs = fit_track5_estimate_calibration(
        _biased_estimates(),
        template=_template(),
        truth=_truth(),
        mode="translation",
    )

    assert config["mode"] == "translation"
    assert config["transform"]["offset_m"] == pytest.approx([-1.5, 2.0, -0.5])
    assert config["fit_summary"]["after_mse_m2"] == pytest.approx(0.0)
    assert pairs["after_error_m"].max() == pytest.approx(0.0)


def test_apply_calibration_preserves_template_rows_and_classification() -> None:
    config, _ = fit_track5_estimate_calibration(
        _biased_estimates(),
        template=_template(),
        truth=_truth(),
        mode="translation",
    )
    calibrated, diagnostics = apply_track5_estimate_calibration(
        _biased_estimates(),
        template=_template(),
        calibration=config,
    )

    assert len(calibrated) == len(_template())
    assert calibrated["state_x_m"].tolist() == pytest.approx([0.0, 1.0, 4.0])
    assert calibrated["state_y_m"].tolist() == pytest.approx([0.0, 0.0, 4.0])
    assert calibrated["classification"].tolist() == [2, 2, 1]
    assert diagnostics["calibrated_row_valid"].all()


def test_write_apply_outputs_produces_leaderboard_ready_zip(tmp_path: Path) -> None:
    config, _ = fit_track5_estimate_calibration(
        _biased_estimates(),
        template=_template(),
        truth=_truth(),
        mode="translation",
    )
    paths = write_track5_estimate_calibration_apply_outputs(
        estimates=_biased_estimates(),
        template=_template(),
        calibration=config,
        output_dir=tmp_path / "apply",
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["valid_calibrated_rows"] == 3
    assert paths["official_zip"].exists()


def test_estimate_calibration_cli_fit_and_apply(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    class_map_csv = tmp_path / "class_map.csv"
    calibration_json = tmp_path / "calibration.json"
    output_dir = tmp_path / "out"
    _biased_estimates().to_csv(estimates_csv, index=False)
    _template().to_csv(template_csv, index=False)
    _truth().to_csv(truth_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = calibration_main(
        [
            "--fit-estimates-csv",
            str(estimates_csv),
            "--apply-estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--truth-csv",
            str(truth_csv),
            "--calibration-json",
            str(calibration_json),
            "--output-dir",
            str(output_dir),
            "--class-map",
            str(class_map_csv),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert calibration_json.exists()
    assert (output_dir / "apply" / "ug2_submission.zip").exists()


def test_estimate_calibration_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-calibration"]
        == "raft_uav.mmuad.track5_estimate_calibration:main"
    )
