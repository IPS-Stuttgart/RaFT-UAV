from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import main as resample_main
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template
from raft_uav.mmuad.track5_template_resample import summarize_template_resample_diagnostics
from raft_uav.mmuad.track5_template_resample import write_track5_template_resample_outputs


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 5.0],
            "state_y_m": [0.0, 20.0, 5.0],
            "state_z_m": [1.0, 3.0, 7.0],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 10.0, 0.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [2, 2, 2, 1],
        }
    )


def test_resample_estimates_to_track5_template_interpolates_midpoints() -> None:
    resampled, diagnostics = resample_estimates_to_track5_template(_estimates(), _template())

    midpoint = resampled.loc[
        (resampled["sequence_id"] == "seq0001") & (resampled["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.0)
    assert midpoint["state_y_m"] == pytest.approx(10.0)
    assert midpoint["state_z_m"] == pytest.approx(2.0)
    assert len(resampled) == 4
    assert diagnostics["valid"].all()


def test_sequence_diagnostics_summarize_invalid_and_extrapolated_rows() -> None:
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001", "seq0003"],
            "Timestamp": [0.0, 5.0, 20.0, 1.0],
        }
    )
    _, diagnostics = resample_estimates_to_track5_template(
        _estimates(),
        template,
        max_nearest_time_delta_s=1.0,
    )

    summary = summarize_template_resample_diagnostics(diagnostics).set_index("sequence_id")

    assert summary.loc["seq0001", "template_row_count"] == 3
    assert summary.loc["seq0001", "valid_row_count"] == 1
    assert summary.loc["seq0001", "invalid_row_count"] == 2
    assert summary.loc["seq0001", "extrapolated_row_count"] == 1
    assert summary.loc["seq0001", "nearest_time_delta_abs_max_s"] == pytest.approx(10.0)
    assert summary.loc["seq0003", "source_row_count_max"] == 0
    assert summary.loc["seq0003", "invalid_row_count"] == 1


def test_write_track5_template_resample_outputs_produces_upload_ready_zip(tmp_path: Path) -> None:
    paths = write_track5_template_resample_outputs(
        estimates=_estimates(),
        template=_template(),
        output_dir=tmp_path,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    assert paths["official_zip"].exists()
    assert paths["official_results_csv"].exists()
    assert paths["diagnostics_by_sequence_csv"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official_rows = pd.read_csv(paths["official_results_csv"])
    assert official_rows["Classification"].tolist() == [2, 2, 2, 1]
    sequence_diagnostics = pd.read_csv(paths["diagnostics_by_sequence_csv"])
    assert set(sequence_diagnostics["sequence_id"]) == {"seq0001", "seq0002"}


def test_template_resample_cli_writes_artifacts(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _estimates().to_csv(estimates_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = resample_main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_template_resample_manifest.json").exists()
    assert (output_dir / "mmuad_template_resample_diagnostics_by_sequence.csv").exists()
    manifest = json.loads((output_dir / "mmuad_template_resample_manifest.json").read_text())
    assert manifest["row_count"] == 4
    assert manifest["sequence_count"] == 2
    assert manifest["invalid_sequence_count"] == 0
    assert manifest["leaderboard_ready"] is True
