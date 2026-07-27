from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def _normalized_submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001", "001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
            "classification": ["1", "2.0"],
        }
    )


@pytest.mark.parametrize("classification", ["1.5", "-1", "4", "true", "not-a-class"])
def test_load_track5_submission_rejects_invalid_normalized_classifications(
    tmp_path: Path,
    classification: str,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    rows = _normalized_submission_rows()
    rows.loc[0, "classification"] = classification
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid Track 5 Classification"):
        load_track5_submission(path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time_s", "not-a-time"),
        ("state_x_m", "nan"),
        ("state_y_m", "inf"),
        ("state_z_m", "-inf"),
    ],
)
def test_load_track5_submission_rejects_nonfinite_normalized_rows(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    rows = _normalized_submission_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = value
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-finite normalized submission values"):
        load_track5_submission(path)
