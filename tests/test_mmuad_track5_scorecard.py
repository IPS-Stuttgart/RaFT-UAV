from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from raft_uav.mmuad.track5_scorecard import (
    build_track5_scorecard,
    scorecard_summary_frame,
    write_track5_scorecard,
)


def _write_official_zip(path: Path, frame: pd.DataFrame) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))
    return path


def test_track5_scorecard_reports_ready_zero_error_submission(tmp_path: Path) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001", "seq001"],
                "Timestamp": [0.0, 1.0],
                "Position": ["(0,0,10)", "(1,0,10)"],
                "Classification": [1, 1],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "class_name": [1, 1],
        }
    ).to_csv(truth_csv, index=False)

    scorecard = build_track5_scorecard(results_path=results_zip, truth_path=truth_csv)

    assert scorecard.summary["validation"]["codabench_upload_ready"] is True
    assert scorecard.summary["scorecard_leaderboard_ready"] is True
    pooled = scorecard.summary["public_track5"]["pooled"]
    assert pooled["pose_mse_loss_m2"] == 0.0
    assert pooled["uav_type_accuracy"] == 1.0
    flat = scorecard_summary_frame(scorecard.summary)
    assert flat.loc[0, "pose_mse_loss_m2"] == 0.0


def test_track5_scorecard_writes_all_requested_artifacts(tmp_path: Path) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001"],
                "Timestamp": [0.0],
                "Position": ["(0,0,10)"],
                "Classification": [2],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [10.0],
            "class_name": [2],
        }
    ).to_csv(truth_csv, index=False)
    scorecard = build_track5_scorecard(results_path=results_zip, truth_path=truth_csv)

    paths = write_track5_scorecard(
        scorecard,
        summary_json=tmp_path / "scorecard.json",
        summary_csv=tmp_path / "scorecard.csv",
        validation_rows_csv=tmp_path / "validation_rows.csv",
        public_evaluation_rows_csv=tmp_path / "public_rows.csv",
        nearest_time_rows_csv=tmp_path / "nearest_rows.csv",
    )

    assert set(paths) == {
        "scorecard_json",
        "scorecard_csv",
        "validation_rows_csv",
        "public_evaluation_rows_csv",
        "nearest_time_rows_csv",
    }
    for path in paths.values():
        assert Path(path).exists()
