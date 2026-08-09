from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_smooth import smooth_track5_submission_rows
from raft_uav.mmuad.track5_trajectory_smooth import write_track5_trajectory_smooth_outputs


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [1.0] * 5,
            "Classification": [2] * 5,
        }
    )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("time_s", np.nan),
        ("time_s", np.inf),
        ("state_x_m", "not-a-number"),
        ("state_y_m", True),
        ("state_z_m", -np.inf),
    ],
)
def test_trajectory_smoother_rejects_rows_legacy_normalizer_would_drop_or_coerce(
    column: str,
    invalid_value: object,
) -> None:
    rows = _rows()
    rows.loc[2, column] = invalid_value

    with pytest.raises(ValueError, match=column):
        smooth_track5_submission_rows(rows)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_s": np.nan}, "window_s"),
        ({"window_s": np.inf}, "window_s"),
        ({"window_s": True}, "window_s"),
        ({"bandwidth_s": np.nan}, "bandwidth_s"),
        ({"bandwidth_s": 0.0}, "bandwidth_s"),
        ({"blend": True}, "blend"),
        ({"blend": np.inf}, "blend"),
        ({"max_correction_m": np.nan}, "max_correction_m"),
        ({"min_neighbors": 0}, "min_neighbors"),
        ({"min_neighbors": 2.5}, "min_neighbors"),
        ({"min_neighbors": True}, "min_neighbors"),
    ],
)
def test_trajectory_smoother_rejects_invalid_controls_without_lossy_coercion(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        smooth_track5_submission_rows(_rows(), **kwargs)


def test_trajectory_smoother_accepts_zero_dimensional_numeric_scalars() -> None:
    smoothed, diagnostics = smooth_track5_submission_rows(
        _rows(),
        window_s=np.array(2.0),
        bandwidth_s=np.array(1.0),
        blend=np.array(0.5),
        max_correction_m=np.array(5.0),
        min_neighbors=np.array(3),
    )

    assert len(smoothed) == 5
    assert len(diagnostics) == 5


def test_trajectory_smoother_writer_fails_before_creating_output_directory(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows.loc[2, "state_x_m"] = np.nan
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="state_x_m"):
        write_track5_trajectory_smooth_outputs(
            rows=rows,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_trajectory_smoother_invalid_control_fails_before_output_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="window_s"):
        write_track5_trajectory_smooth_outputs(
            rows=_rows(),
            output_dir=output_dir,
            window_s=np.nan,
        )

    assert not output_dir.exists()
