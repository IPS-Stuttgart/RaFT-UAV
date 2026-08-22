from __future__ import annotations

import json

import pytest

from raft_uav.multi_uav_lts._records import Detection, parse_detection_text
from raft_uav.multi_uav_lts.trajectory_box_calibration import (
    BoxCalibrationParameters,
    _parameters_for_sequence,
    calibrate_prediction_set,
    calibrate_sequence,
)


def _row(frame: int, center_x: float, *, confidence: float = 1.0) -> Detection:
    return Detection(
        frame,
        7,
        center_x - 5.0,
        20.0,
        10.0,
        8.0,
        confidence,
        1,
        1.0,
    )


def test_rts_smoothing_reduces_isolated_center_outlier() -> None:
    rows = (
        _row(1, 10.0),
        _row(2, 11.0),
        _row(3, 24.0, confidence=0.2),
        _row(4, 13.0),
        _row(5, 14.0),
    )
    parameters = BoxCalibrationParameters(
        process_accel_sigma=0.02,
        center_measurement_sigma=0.45,
        recenter_weight=1.0,
        size_smoothing_weight=0.0,
    )
    calibrated, summary = calibrate_sequence(
        "BB1_00",
        rows,
        parameters=parameters,
    )
    frame_three = next(row for row in calibrated if row.frame_id == 3)
    assert abs(frame_three.center_x - 12.0) < abs(24.0 - 12.0)
    assert summary.smoothed_track_count == 1
    assert {(row.frame_id, row.object_id) for row in calibrated} == {
        (row.frame_id, row.object_id) for row in rows
    }


def test_uncertainty_expansion_is_bounded_and_preserves_metadata() -> None:
    rows = (_row(1, 10.0, confidence=0.05), _row(3, 12.0, confidence=0.05))
    parameters = BoxCalibrationParameters(
        process_accel_sigma=0.5,
        center_measurement_sigma=0.5,
        uncertainty_scale_x=4.0,
        uncertainty_scale_y=4.0,
        max_area_ratio=1.5,
        image_width=30.0,
        image_height=30.0,
    )
    calibrated, summary = calibrate_sequence("T_00", rows, parameters=parameters)
    assert summary.max_area_ratio <= pytest.approx(1.5, abs=1e-10)
    for original, output in zip(rows, calibrated):
        assert output.frame_id == original.frame_id
        assert output.object_id == original.object_id
        assert output.class_id == original.class_id
        assert output.visibility == original.visibility
        assert output.confidence == original.confidence
        assert output.x1 >= 0.0
        assert output.y1 >= 0.0
        assert output.x1 + output.width <= 30.0 + 1e-9
        assert output.y1 + output.height <= 30.0 + 1e-9


def test_policy_uses_longest_matching_prefix() -> None:
    base = BoxCalibrationParameters()
    policy = {
        "schema": "raft-uav-multi-uav-lts-box-calibration-policy-v1",
        "default": {"uncertainty_scale_x": 0.25},
        "prefixes": {
            "BB2": {"uncertainty_scale_x": 0.5},
            "BB2P": {"uncertainty_scale_x": 1.25, "uncertainty_scale_y": 0.75},
        },
    }
    selected = _parameters_for_sequence(base, "BB2P_02", policy)
    assert selected.uncertainty_scale_x == 1.25
    assert selected.uncertainty_scale_y == 0.75


def test_prediction_set_round_trip_and_policy(tmp_path) -> None:
    predictions = tmp_path / "predictions"
    output = tmp_path / "calibrated"
    predictions.mkdir()
    (predictions / "BB2P_02.txt").write_text(
        "1,7,5,20,10,8,0.8,1,1\n2,7,6,20,10,8,0.8,1,1\n",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema": "raft-uav-multi-uav-lts-box-calibration-policy-v1",
                "default": {"size_smoothing_weight": 0.0},
                "prefixes": {"BB2P": {"uncertainty_scale_x": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    summary = calibrate_prediction_set(
        predictions,
        output,
        policy_path=policy_path,
    )
    rows = parse_detection_text(
        (output / "BB2P_02.txt").read_text(encoding="utf-8"),
        source="round-trip",
    )
    assert summary.sequence_count == 1
    assert summary.row_count == 2
    assert [(row.frame_id, row.object_id) for row in rows] == [(1, 7), (2, 7)]
