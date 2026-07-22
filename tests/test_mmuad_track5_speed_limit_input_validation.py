from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.track5_speed_limit as speed_limit
from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 100.0, 200.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    "iterations",
    [0, -1, 1.5, True, False, np.nan, np.inf, -np.inf],
)
def test_speed_limit_rejects_invalid_iteration_counts(iterations: object) -> None:
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        project_track5_speed_limit(_submission(), iterations=iterations)


def test_speed_limit_accepts_integer_equivalent_iteration_count() -> None:
    expected, expected_diagnostics = project_track5_speed_limit(
        _submission(),
        max_speed_mps=10.0,
        iterations=1,
    )
    actual, actual_diagnostics = project_track5_speed_limit(
        _submission(),
        max_speed_mps=10.0,
        iterations=1.0,
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_diagnostics, expected_diagnostics)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time_s", np.nan),
        ("state_x_m", np.inf),
        ("state_y_m", -np.inf),
        ("state_z_m", "not-a-number"),
        ("Classification", pd.NA),
    ],
)
def test_speed_limit_rejects_rows_with_invalid_numeric_values(
    column: str,
    value: object,
) -> None:
    rows = _submission()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value

    with pytest.raises(ValueError) as error:
        project_track5_speed_limit(rows)

    assert "submission contains non-finite numeric values" in str(error.value)
    assert f"{column} rows [1]" in str(error.value)


def test_speed_limit_reports_every_invalid_numeric_column() -> None:
    rows = _submission()
    rows.loc[0, "time_s"] = np.nan
    rows.loc[2, "state_z_m"] = np.inf

    with pytest.raises(ValueError) as error:
        project_track5_speed_limit(rows)

    message = str(error.value)
    assert "time_s rows [0]" in message
    assert "state_z_m rows [2]" in message


def test_speed_limit_rejects_duplicate_fixed_grid_keys() -> None:
    rows = _submission()
    rows.loc[2, "time_s"] = 1.0
    rows.loc[2, "state_x_m"] = 150.0

    with pytest.raises(ValueError) as error:
        project_track5_speed_limit(rows, max_speed_mps=10.0)

    message = str(error.value)
    assert "1 duplicate (sequence_id, time_s) key(s)" in message
    assert "seq0001@1" in message


def test_speed_limit_rejects_numeric_equivalent_duplicate_timestamps() -> None:
    rows = _submission()
    rows["time_s"] = rows["time_s"].astype(object)
    rows.loc[2, "time_s"] = "1"

    with pytest.raises(ValueError, match=r"seq0001@1"):
        project_track5_speed_limit(rows)


def test_speed_limit_allows_timestamp_reuse_across_sequences() -> None:
    rows = _submission()
    rows.loc[2, "sequence_id"] = "seq0002"
    rows.loc[2, "time_s"] = 1.0

    limited, diagnostics = project_track5_speed_limit(rows, max_speed_mps=10.0)

    assert len(limited) == len(rows)
    assert len(diagnostics) == len(rows)
    assert set(limited["sequence_id"]) == {"seq0001", "seq0002"}


def test_speed_limit_cli_resolves_validated_public_projector() -> None:
    assert speed_limit.main.__globals__["project_track5_speed_limit"] is project_track5_speed_limit
