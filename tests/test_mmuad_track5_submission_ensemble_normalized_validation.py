from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def _normalized_submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001", "001", "002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [0.0, 2.0, 10.0],
            "state_y_m": [0.0, 0.0, 1.0],
            "state_z_m": [1.0, 1.0, 2.0],
            "classification": [1, 1, 2],
        }
    )


@pytest.mark.parametrize(
    ("column", "value", "expected_field"),
    [
        ("time_s", np.nan, "time_s"),
        ("state_x_m", np.inf, "state_x_m"),
        ("state_y_m", -np.inf, "state_y_m"),
        ("state_z_m", "not-a-number", "state_z_m"),
        ("classification", np.nan, "Classification"),
    ],
)
def test_load_track5_submission_rejects_invalid_normalized_numeric_rows(
    tmp_path: Path,
    column: str,
    value: object,
    expected_field: str,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    rows = _normalized_submission_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match=rf"invalid normalized {expected_field} row") as exc_info:
        load_track5_submission(path)

    assert "index/indices 1" in str(exc_info.value)


def test_load_track5_submission_rejects_fractional_normalized_classification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    rows = _normalized_submission_rows()
    rows["classification"] = rows["classification"].astype(object)
    rows.loc[0, "classification"] = "1.5"
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid normalized Classification values"):
        load_track5_submission(path)
