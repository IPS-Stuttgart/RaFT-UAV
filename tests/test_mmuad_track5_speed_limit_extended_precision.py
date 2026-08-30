from __future__ import annotations

import numpy as np
import pandas as pd

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


def _nested_object_array(value: object) -> np.ndarray:
    nested = np.empty((), dtype=object)
    nested[()] = value
    return nested


def test_speed_limit_accepts_extended_precision_controls() -> None:
    expected, expected_diagnostics = project_track5_speed_limit(
        _submission(),
        max_speed_mps=10.0,
        iterations=2,
        anchor_blend=0.25,
    )

    actual, actual_diagnostics = project_track5_speed_limit(
        _submission(),
        max_speed_mps=_nested_object_array(np.longdouble("10.0")),
        iterations=_nested_object_array(np.longdouble("2.0")),
        anchor_blend=_nested_object_array(np.longdouble("0.25")),
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_diagnostics, expected_diagnostics)


def test_speed_limit_accepts_extended_precision_submission_cells() -> None:
    expected, expected_diagnostics = project_track5_speed_limit(
        _submission(),
        max_speed_mps=10.0,
    )
    rows = _submission().astype(
        {
            "time_s": object,
            "state_x_m": object,
            "state_y_m": object,
            "state_z_m": object,
            "Classification": object,
        }
    )
    rows.at[1, "time_s"] = np.longdouble("1.0")
    rows.at[1, "state_x_m"] = np.longdouble("100.0")
    rows.at[1, "state_y_m"] = np.longdouble("0.0")
    rows.at[1, "state_z_m"] = np.longdouble("0.0")
    rows.at[1, "Classification"] = np.longdouble("2.0")

    actual, actual_diagnostics = project_track5_speed_limit(
        rows,
        max_speed_mps=np.longdouble("10.0"),
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_diagnostics, expected_diagnostics)
