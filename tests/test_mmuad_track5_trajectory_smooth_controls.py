from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_smooth import (
    smooth_track5_submission_rows,
    write_track5_trajectory_smooth_outputs,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["sequence"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [1, 1, 1],
        }
    )


@pytest.mark.parametrize(
    ("control", "value"),
    [
        pytest.param("window_s", True, id="boolean-window"),
        pytest.param("window_s", np.inf, id="infinite-window"),
        pytest.param("bandwidth_s", np.nan, id="nan-bandwidth"),
        pytest.param("blend", np.bool_(True), id="numpy-boolean-blend"),
        pytest.param("blend", np.complex128(0.5 + 0.0j), id="complex-blend"),
        pytest.param(
            "max_correction_m",
            np.asarray(True, dtype=object),
            id="boxed-boolean-correction-cap",
        ),
        pytest.param("min_neighbors", 2.5, id="fractional-neighbor-count"),
        pytest.param("min_neighbors", 0, id="zero-neighbor-count"),
    ],
)
def test_trajectory_smoother_rejects_lossy_controls(control: str, value: object) -> None:
    with pytest.raises(ValueError):
        smooth_track5_submission_rows(_rows(), **{control: value})


def test_write_path_uses_strict_smoother_control_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="min_neighbors"):
        write_track5_trajectory_smooth_outputs(
            rows=_rows(),
            output_dir=tmp_path,
            min_neighbors=np.asarray(1.5),
        )

    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("time_s", "not-a-time", id="nonnumeric-time"),
        pytest.param("state_x_m", np.inf, id="infinite-position"),
        pytest.param("state_y_m", True, id="boolean-position"),
        pytest.param("state_z_m", np.complex128(1.0 + 2.0j), id="complex-position"),
    ],
)
def test_trajectory_smoother_rejects_invalid_fixed_grid_numeric_rows(
    column: str,
    value: object,
) -> None:
    rows = _rows().astype(object)
    rows.index = [10, 20, 30]
    rows.loc[20, column] = value

    with pytest.raises(ValueError, match=rf"{column}.*20"):
        smooth_track5_submission_rows(rows)


@pytest.mark.parametrize("value", [None, "", "   ", np.nan])
def test_trajectory_smoother_rejects_invalid_sequence_ids(value: object) -> None:
    rows = _rows().astype(object)
    rows.index = [10, 20, 30]
    rows.loc[20, "sequence_id"] = value

    with pytest.raises(ValueError, match=r"sequence_id.*20"):
        smooth_track5_submission_rows(rows)


@pytest.mark.parametrize("value", [None, 4, 1.5, True])
def test_trajectory_smoother_rejects_invalid_classifications(value: object) -> None:
    rows = _rows().astype(object)
    rows.index = [10, 20, 30]
    rows.loc[20, "Classification"] = value

    with pytest.raises(ValueError, match=r"Classification.*20"):
        smooth_track5_submission_rows(rows)


def test_trajectory_smoother_preserves_valid_row_count_and_canonicalizes_cells() -> None:
    rows = _rows().astype(object)
    rows["sequence_id"] = [" 001 ", " 001 ", " 001 "]
    rows["time_s"] = ["0", "1.0", np.asarray(2.0)]
    rows["Classification"] = ["1", "1", np.int64(1)]

    smoothed, diagnostics = smooth_track5_submission_rows(rows)

    assert len(smoothed) == len(rows)
    assert len(diagnostics) == len(rows)
    assert set(smoothed["sequence_id"]) == {"001"}
    assert smoothed["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert smoothed["Classification"].tolist() == [1, 1, 1]
