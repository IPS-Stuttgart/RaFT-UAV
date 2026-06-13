from zipfile import ZipFile

import pandas as pd

from raft_uav.mmuad.submission import validate_official_track5_submission


def test_official_track5_submission_validator_rejects_invalid_sequence_and_timestamp(
    tmp_path,
):
    zip_path = tmp_path / "invalid_rows.zip"
    frame = pd.DataFrame(
        {
            "Sequence": ["   ", "seq1"],
            "Timestamp": [0.0, float("nan")],
            "Position": ["(0,0,0)", "(1,1,1)"],
            "Classification": [0, 0],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))

    validation = validate_official_track5_submission(zip_path)

    assert validation.summary["valid"] is False
    assert validation.summary["invalid_sequence_count"] == 1
    assert validation.summary["invalid_timestamp_count"] == 1
    assert "invalid_sequence" in validation.rows["status"].tolist()
    assert "invalid_timestamp" in validation.rows["status"].tolist()
