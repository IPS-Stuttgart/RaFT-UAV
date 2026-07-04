from __future__ import annotations

from zipfile import ZipFile

import pandas as pd

from raft_uav.mmuad.submission import (
    OFFICIAL_UG2_RESULT_COLUMNS,
    validate_official_track5_submission,
)


def test_official_track5_validator_handles_invalid_only_submission_with_template(
    tmp_path,
) -> None:
    zip_path = tmp_path / "invalid_only.zip"
    invalid_results = pd.DataFrame(
        {
            "Sequence": [" "],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", invalid_results.to_csv(index=False))

    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [0.0]})
    validation = validate_official_track5_submission(zip_path, template=template)

    assert validation.summary["valid"] is False
    assert validation.summary["invalid_sequence_count"] == 1
    assert validation.summary["template_timestamp_count"] == 1
    assert validation.summary["missing_template_timestamp_count"] == 1
    assert "official_invalid_rows" in validation.summary["leaderboard_blocking_reasons"]
    assert (
        "official_missing_template_timestamps"
        in validation.summary["leaderboard_blocking_reasons"]
    )


def test_official_track5_validator_handles_empty_results_with_template(tmp_path) -> None:
    zip_path = tmp_path / "empty_results.zip"
    empty_results = pd.DataFrame(columns=OFFICIAL_UG2_RESULT_COLUMNS)
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", empty_results.to_csv(index=False))

    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [0.0]})
    validation = validate_official_track5_submission(zip_path, template=template)

    assert validation.summary["valid"] is False
    assert validation.summary["row_count"] == 0
    assert validation.summary["template_timestamp_count"] == 1
    assert validation.summary["missing_template_timestamp_count"] == 1
    assert validation.summary["duplicate_prediction_count"] == 0
    assert validation.summary["extra_prediction_count"] == 0
    assert validation.rows["status"].tolist() == ["missing_template_timestamp"]
