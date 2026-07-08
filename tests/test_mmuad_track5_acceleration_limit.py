from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import main as acceleration_main
from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks
from raft_uav.mmuad.track5_acceleration_limit import write_track5_acceleration_limit_outputs


def _kink_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 2.0, 0.0],
            "state_x_m": [0.0, 10.0, 2.0, 5.0],
            "state_y_m": [0.0, 0.0, 0.0, 5.0],
            "state_z_m": [0.0, 0.0, 0.0, 5.0],
            "Classification": [2, 2, 2, 1],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 2.0, 0.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [2, 2, 2, 1],
        }
    )


def test_acceleration_limit_repairs_high_acceleration_kink_without_speed_violation() -> None:
    repaired, diagnostics = repair_track5_acceleration_kinks(
        _kink_submission(),
        max_acceleration_mps2=5.0,
        max_direct_speed_mps=20.0,
        min_interpolation_residual_m=1.0,
        iterations=1,
    )

    midpoint = repaired.loc[(repaired["sequence_id"] == "seq0001") & (repaired["time_s"] == 1.0)].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(1.0)
    assert midpoint["state_y_m"] == pytest.approx(0.0)
    assert midpoint["Classification"] == 2
    repaired_row = diagnostics.loc[
        (diagnostics["sequence_id"] == "seq0001") & (diagnostics["time_s"] == 1.0)
    ].iloc[0]
    assert repaired_row["acceleration_limit_applied"]
    assert repaired_row["local_acceleration_mps2"] > 5.0
    assert repaired_row["neighbor_direct_speed_mps"] <= 20.0


def test_acceleration_limit_blend_can_apply_partial_repair() -> None:
    repaired, _ = repair_track5_acceleration_kinks(
        _kink_submission(),
        max_acceleration_mps2=5.0,
        max_direct_speed_mps=20.0,
        min_interpolation_residual_m=1.0,
        iterations=1,
        repair_blend=0.5,
    )

    midpoint = repaired.loc[(repaired["sequence_id"] == "seq0001") & (repaired["time_s"] == 1.0)].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.5)


def test_acceleration_limit_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    repaired, diagnostics = repair_track5_acceleration_kinks(
        _kink_submission(),
        max_acceleration_mps2=5.0,
        max_direct_speed_mps=20.0,
        min_interpolation_residual_m=1.0,
        iterations=1,
    )
    paths = write_track5_acceleration_limit_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
        template=_template(),
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["changed_row_count"] == 1
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_acceleration_limit_cli_writes_outputs(tmp_path: Path) -> None:
    submission_csv = tmp_path / "submission.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _kink_submission().to_csv(submission_csv, index=False)
    _template().to_csv(template_csv, index=False)

    status = acceleration_main(
        [
            "--submission",
            str(submission_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--max-acceleration-mps2",
            "5",
            "--max-direct-speed-mps",
            "20",
            "--min-interpolation-residual-m",
            "1",
            "--iterations",
            "1",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission_acceleration_limited.zip").exists()
    assert (output_dir / "mmuad_track5_acceleration_limit_manifest.json").exists()
