import numpy as np

from raft_uav.baselines.imm import run_async_imm_baseline
from raft_uav.baselines.kalman import TrackingMeasurement


def _same_timestamp_measurements() -> tuple[TrackingMeasurement, TrackingMeasurement]:
    rf = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([10.0, 20.0]),
        covariance=np.diag([4.0, 4.0]),
        source="rf",
    )
    radar = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([30.0, 40.0, 50.0]),
        covariance=np.diag([9.0, 9.0, 16.0]),
        source="radar",
    )
    return rf, radar


def test_async_imm_same_timestamp_order_is_input_independent() -> None:
    rf, radar = _same_timestamp_measurements()

    forward = run_async_imm_baseline(
        [rf, radar],
        acceleration_std_mps2=0.0,
    )
    reversed_input = run_async_imm_baseline(
        [radar, rf],
        acceleration_std_mps2=0.0,
    )

    assert [row["source"] for row in forward] == ["rf", "radar"]
    assert [row["source"] for row in reversed_input] == ["rf", "radar"]
    assert [row["update_action"] for row in forward] == [
        row["update_action"] for row in reversed_input
    ]
    for expected, actual in zip(forward, reversed_input, strict=True):
        np.testing.assert_allclose(actual["state"], expected["state"])
        np.testing.assert_allclose(actual["covariance"], expected["covariance"])
        np.testing.assert_allclose(
            actual["mode_probabilities"],
            expected["mode_probabilities"],
        )
        assert actual["most_likely_mode"] == expected["most_likely_mode"]
