from __future__ import annotations

import pytest

from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.trajectory_box_calibration import BoxCalibrationParameters
from raft_uav.multi_uav_lts.trajectory_gap_completion import (
    GapCompletionParameters,
    complete_sequence,
)


def _row(frame: int, center_x: float, *, confidence: float = 0.9) -> Detection:
    return Detection(
        frame,
        3,
        center_x - 2.0,
        8.0,
        4.0,
        4.0,
        confidence,
        1,
        1.0,
    )


def test_two_sided_bridge_fills_short_internal_gap() -> None:
    rows = (_row(1, 10.0), _row(4, 13.0))
    completed, summary = complete_sequence(
        "C_00",
        rows,
        parameters=GapCompletionParameters(
            max_gap_frames=2,
            use_smoothed_endpoints=False,
        ),
        smoother_parameters=BoxCalibrationParameters(),
    )

    assert [(row.frame_id, row.object_id) for row in completed] == [
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
    ]
    assert completed[1].center_x == pytest.approx(11.0)
    assert completed[2].center_x == pytest.approx(12.0)
    assert summary.inserted_rows == 2
    assert summary.completed_gaps == 1


def test_gap_completion_rejects_implausible_motion() -> None:
    rows = (_row(1, 10.0), _row(3, 200.0))
    completed, summary = complete_sequence(
        "T_00",
        rows,
        parameters=GapCompletionParameters(
            max_gap_frames=2,
            max_normalized_speed=2.0,
        ),
        smoother_parameters=BoxCalibrationParameters(),
    )

    assert completed == rows
    assert summary.inserted_rows == 0
    assert summary.rejected_motion_gaps == 1


def test_gap_completion_rejects_low_confidence_endpoints() -> None:
    rows = (_row(1, 10.0, confidence=0.001), _row(3, 12.0))
    completed, summary = complete_sequence(
        "TF_00",
        rows,
        parameters=GapCompletionParameters(
            max_gap_frames=2,
            min_endpoint_confidence=0.01,
        ),
        smoother_parameters=BoxCalibrationParameters(),
    )

    assert completed == rows
    assert summary.rejected_confidence_gaps == 1
