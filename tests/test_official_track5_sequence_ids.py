from __future__ import annotations

from zipfile import ZipFile

import pandas as pd

from raft_uav.mmuad.submission import (
    load_official_track5_results_frame,
    load_official_track5_template_file,
    validate_official_track5_submission,
)


def _official_rows(sequence_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": sequence_ids,
            "Timestamp": [float(index) for index, _ in enumerate(sequence_ids)],
            "Position": [f"({index},0,0)" for index, _ in enumerate(sequence_ids)],
            "Classification": [2 for _ in sequence_ids],
        }
    )


def test_track5_results_loader_preserves_zero_padded_csv_sequence_ids(tmp_path):
    csv_path = tmp_path / "mmaud_results.csv"
    _official_rows(["001", "010"]).to_csv(csv_path, index=False)

    loaded = load_official_track5_results_frame(csv_path)

    assert loaded["Sequence"].tolist() == ["001", "010"]


def test_track5_results_loader_preserves_zero_padded_zip_sequence_ids(tmp_path):
    zip_path = tmp_path / "submission.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", _official_rows(["001", "010"]).to_csv(index=False))

    loaded = load_official_track5_results_frame(zip_path)

    assert loaded["Sequence"].tolist() == ["001", "010"]


def test_track5_template_loader_preserves_zero_padded_sequence_ids(tmp_path):
    template_path = tmp_path / "template.csv"
    _official_rows(["001", "010"]).to_csv(template_path, index=False)

    template = load_official_track5_template_file(template_path)

    assert template["sequence_id"].tolist() == ["001", "010"]


def test_track5_submission_validator_matches_zero_padded_template_sequence_ids(tmp_path):
    zip_path = tmp_path / "submission.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", _official_rows(["001"]).to_csv(index=False))
    template = pd.DataFrame({"sequence_id": ["001"], "time_s": [0.0]})

    validation = validate_official_track5_submission(zip_path, template=template)

    assert validation.summary["leaderboard_ready"] is True
    assert validation.summary["missing_template_timestamp_count"] == 0
    assert validation.summary["extra_prediction_count"] == 0
    assert set(validation.summary["sequences"]) == {"001"}
