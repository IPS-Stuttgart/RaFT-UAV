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
    ("column", "value"),
    [
        ("time_s", np.nan),
        ("state_x_m", np.inf),
        ("state_y_m", -np.inf),
        ("state_z_m", "not-a-number"),
    ],
)
def test_load_track5_submission_rejects_malformed_normalized_measurements(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    rows = _normalized_submission_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value
    rows.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match=rf"invalid normalized Track 5 {column} row",
    ) as exc_info:
        load_track5_submission(path)

    assert "index/indices 1" in str(exc_info.value)


def test_load_track5_submission_preserves_all_valid_normalized_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "normalized_submission.csv"
    _normalized_submission_rows().to_csv(path, index=False)

    loaded = load_track5_submission(path)

    assert len(loaded) == 3
    assert loaded[["sequence_id", "time_s"]].to_records(index=False).tolist() == [
        ("001", 0.0),
        ("001", 1.0),
        ("002", 0.0),
    ]
