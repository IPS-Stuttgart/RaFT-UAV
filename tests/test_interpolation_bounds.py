import numpy as np

from raft_uav.baselines.radar_association import _within_interpolation_gap
from raft_uav.baselines.radar_association import _within_interpolation_speed


def test_gap_mask_does_not_extrapolate_past_anchors():
    frame_times = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    anchor_times = np.array([0.0, 1.0, 2.0])

    kept = _within_interpolation_gap(frame_times, anchor_times, max_gap_s=1.1)

    np.testing.assert_array_equal(
        kept,
        [False, True, True, True, True, True, False],
    )


def test_speed_mask_does_not_extrapolate_past_anchors():
    frame_times = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    anchor_times = np.array([0.0, 1.0, 2.0])
    anchor_positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )

    kept = _within_interpolation_speed(
        frame_times,
        anchor_times,
        anchor_positions,
        max_speed_mps=2.0,
    )

    np.testing.assert_array_equal(
        kept,
        [False, True, True, True, True, True, False],
    )


def test_gap_mask_requires_strict_anchor_match_at_large_timestamps():
    frame_times = np.array([1000.0, 1001.995, 1002.0])
    anchor_times = np.array([1000.0, 1002.0])

    kept = _within_interpolation_gap(frame_times, anchor_times, max_gap_s=1.0)

    np.testing.assert_array_equal(kept, [True, False, True])


def test_speed_mask_requires_strict_anchor_match_at_large_timestamps():
    frame_times = np.array([1000.0, 1001.995, 1002.0])
    anchor_times = np.array([1000.0, 1002.0])
    anchor_positions = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    kept = _within_interpolation_speed(
        frame_times,
        anchor_times,
        anchor_positions,
        max_speed_mps=10.0,
    )

    np.testing.assert_array_equal(kept, [True, False, True])
