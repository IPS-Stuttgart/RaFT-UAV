from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard_cli import main as track5_scorecard_cli_main
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


def test_track5_scorecard_cli_writes_ready_artifacts(tmp_path: Path, capsys) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001", "seq001"],
                "Timestamp": [0.0, 1.0],
                "Position": ["(0,0,10)", "(1,0,10)"],
                "Classification": [2, 2],
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
            "class_name": [2, 2],
        }
    ).to_csv(truth_csv, index=False)
    output = tmp_path / "scorecard"

    status = track5_scorecard_cli_main(
        [
            "--results",
            str(results_zip),
            "--truth",
            str(truth_csv),
            "--output-json",
            str(output / "scorecard.json"),
            "--summary-csv",
            str(output / "scorecard.csv"),
            "--validation-rows-csv",
            str(output / "validation_rows.csv"),
            "--public-evaluation-rows-csv",
            str(output / "public_rows.csv"),
            "--nearest-time-rows-csv",
            str(output / "nearest_rows.csv"),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    stdout = capsys.readouterr().out
    assert "track5_scorecard=ok" in stdout
    assert "leaderboard_ready=True" in stdout
    summary = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "raft-uav-mmuad-track5-scorecard-v1"
    assert summary["scorecard_leaderboard_ready"] is True
    assert summary["codabench_upload_ready"] is True
    assert summary["public_track5"]["pooled"]["pose_mse_loss_m2"] == 0.0
    assert (output / "scorecard.csv").exists()
    assert (output / "validation_rows.csv").exists()
    assert (output / "public_rows.csv").exists()
    assert (output / "nearest_rows.csv").exists()


def test_track5_scorecard_cli_requires_truth_for_leaderboard_ready(
    tmp_path: Path,
) -> None:
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
    template_csv = tmp_path / "template.csv"
    pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    ).to_csv(template_csv, index=False)
    output_json = tmp_path / "scorecard.json"

    with pytest.raises(SystemExit, match="public_track5_evaluation_not_run"):
        track5_scorecard_cli_main(
            [
                "--results",
                str(results_zip),
                "--template",
                str(template_csv),
                "--output-json",
                str(output_json),
                "--require-leaderboard-ready",
            ]
        )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["validation"]["codabench_upload_ready"] is True
    assert summary["scorecard_leaderboard_ready"] is False
    assert summary["leaderboard_blocking_reasons"] == [
        "public_track5_evaluation_not_run"
    ]
