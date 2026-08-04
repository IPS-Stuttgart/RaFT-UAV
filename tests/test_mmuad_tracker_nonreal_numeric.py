from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker, select_tracklet_path


_INVALID_REQUIRED_SCALARS = (
    True,
    np.bool_(False),
    1.0 + 2.0j,
    np.complex128(2.0 + 3.0j),
)


@pytest.mark.parametrize("column", ["time_s", "x_m", "y_m", "z_m"])
@pytest.mark.parametrize("invalid_value", _INVALID_REQUIRED_SCALARS)
def test_tracker_ignores_non_real_required_candidate_values(
    column: str,
    invalid_value: object,
) -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [1.0, 2.0],
            "source": ["radar", "radar"],
            "x_m": [1.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        },
        dtype=object,
    )
    rows.at[0, column] = invalid_value

    output = run_mmuad_tracker(
        CandidateFrame(rows),
        config=TrackerConfig(selection_mobility_radius_m=0.0),
    )

    assert output.selected_tracklets["time_s"].tolist() == [2.0]
    assert output.estimates["time_s"].tolist() == [2.0]


def test_tracker_does_not_treat_boolean_confidence_as_one() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [1.0, 1.0],
            "source": ["radar", "radar"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [True, 0.5],
        }
    )

    selected = select_tracklet_path(
        candidates,
        config=TrackerConfig(selection_mobility_radius_m=0.0),
    )

    assert selected["x_m"].tolist() == [100.0]


def test_tracker_preserves_zero_dimensional_numeric_scalars() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [np.array(1.0), np.array(2.0)],
            "source": ["radar", "radar"],
            "x_m": [np.array(1.0), np.array(2.0)],
            "y_m": [np.array(0.0), np.array(0.0)],
            "z_m": [np.array(0.0), np.array(0.0)],
        }
    )

    output = run_mmuad_tracker(
        CandidateFrame(rows),
        config=TrackerConfig(selection_mobility_radius_m=0.0),
    )

    assert output.selected_tracklets["time_s"].tolist() == [1.0, 2.0]
    assert output.estimates["time_s"].tolist() == [1.0, 2.0]
