from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad import submission


def _official_frame(classification: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(1.0,2.0,3.0)"],
            "Classification": [classification],
        }
    )


def test_validate_official_track5_submission_rejects_out_of_domain_classification(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "mmaud_results.csv"
    _official_frame(4).to_csv(results_path, index=False)

    validation = submission.validate_official_track5_submission(
        results_path,
        require_zip=False,
    )

    prediction_rows = validation.rows.loc[validation.rows["row_type"] == "prediction"]
    assert validation.summary["invalid_classification_count"] == 1
    assert not validation.summary["valid"]
    assert prediction_rows["status"].tolist() == ["invalid_classification"]
    assert "must be one of {0, 1, 2, 3}" in prediction_rows["reason"].iloc[0]


def test_normalize_official_track5_results_frame_rejects_out_of_domain_classification() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        submission.normalize_official_track5_results_frame(_official_frame(4))
