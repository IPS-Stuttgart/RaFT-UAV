from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.paper_strict import (
    PAPER_STRICT_LW1_ORIGIN_LLA_ENV,
    PAPER_STRICT_NIS_GATE_PROBABILITY,
    _handle_count_mismatch,
    _projector_for_origin,
    build_count_audit,
    require_fortem_range_m,
    paper_strict_range_gated_radar_candidates,
    select_paper_strict_radar_track,
)
from raft_uav.evaluation.metrics import (
    empirical_position_covariance_at_times,
    position_errors_at_times_m,
)
from raft_uav.baselines.kalman import gate_threshold_from_probability


def test_position_errors_at_times_interpolates_truth_to_measurement_time() -> None:
    errors = position_errors_at_times_m(
        estimate_times_s=np.array([5.0]),
        estimate_positions_m=np.array([[6.0, 0.0, 0.0]]),
        truth_times_s=np.array([0.0, 10.0]),
        truth_positions_m=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        dimensions=3,
    )

    assert np.allclose(errors, np.array([1.0]))


def test_position_errors_at_times_preserves_duplicate_estimate_timestamps() -> None:
    errors = position_errors_at_times_m(
        estimate_times_s=np.array([5.0, 5.0]),
        estimate_positions_m=np.array([[5.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        truth_times_s=np.array([0.0, 10.0]),
        truth_positions_m=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        dimensions=3,
    )

    assert errors.shape == (2,)
    assert np.allclose(errors, np.array([0.0, 2.0]))


def test_empirical_covariance_uses_residuals_at_measurement_times() -> None:
    covariance = empirical_position_covariance_at_times(
        estimate_times_s=np.array([0.0, 1.0, 2.0]),
        estimate_positions_m=np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0]]),
        truth_times_s=np.array([0.0, 1.0, 2.0]),
        truth_positions_m=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        dimensions=2,
    )

    residuals = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    assert np.allclose(covariance, np.cov(residuals, rowvar=False, ddof=1))


def test_empirical_covariance_preserves_duplicate_estimate_timestamps() -> None:
    covariance = empirical_position_covariance_at_times(
        estimate_times_s=np.array([1.0, 1.0, 2.0]),
        estimate_positions_m=np.array([[2.0, 0.0], [4.0, 0.0], [5.0, 1.0]]),
        truth_times_s=np.array([0.0, 1.0, 2.0]),
        truth_positions_m=np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 1.0]]),
        dimensions=2,
    )

    residuals = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
    assert np.allclose(covariance, np.cov(residuals, rowvar=False, ddof=1))


def test_paper_strict_gate_probability_is_95_percent() -> None:
    assert PAPER_STRICT_NIS_GATE_PROBABILITY == pytest.approx(0.95)
    assert gate_threshold_from_probability(PAPER_STRICT_NIS_GATE_PROBABILITY, 2) == pytest.approx(
        5.991464547107979
    )
    assert gate_threshold_from_probability(PAPER_STRICT_NIS_GATE_PROBABILITY, 3) == pytest.approx(
        7.814727903251179
    )


def test_strict_range_gate_requires_fortem_range_column() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "frame_index": [0],
            "track_id": [1],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    with pytest.raises(ValueError, match="range_m"):
        require_fortem_range_m(radar)


def test_select_paper_strict_radar_track_largest_continuous_segment() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 0.0, 1.0, 2.0],
            "frame_index": [0, 1, 0, 1, 2],
            "track_id": [1, 1, 2, 2, 2],
            "east_m": [0.0, 1.0, 0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 1.0, 1.0, 1.0],
            "up_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "range_m": [100.0, 100.0, 100.0, 100.0, 100.0],
            "cat_prob_uav": [0.5, 0.5, 0.4, 0.4, 0.4],
        }
    )

    selected = select_paper_strict_radar_track(radar, range_gate_m=800.0)

    assert selected["track_id"].astype(int).unique().tolist() == [2]
    assert len(selected) == 3


def test_paper_strict_range_gated_candidates_use_fortem_range() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "frame_index": [0, 1],
            "track_id": [1, 1],
            # ENU norm is below 800 m for both rows, but Fortem range excludes
            # the second.  Paper parity must use the native radar range.
            "east_m": [1.0, 1.0],
            "north_m": [1.0, 1.0],
            "up_m": [1.0, 1.0],
            "range_m": [100.0, 900.0],
            "cat_prob_uav": [0.5, 0.5],
        }
    )

    selected = paper_strict_range_gated_radar_candidates(
        radar,
        range_gate_m=800.0,
        require_range_m=True,
    )

    assert len(selected) == 1
    assert selected["frame_index"].tolist() == [0]


def test_lw1_origin_can_be_supplied_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PAPER_STRICT_LW1_ORIGIN_LLA_ENV, "35.0,-78.0,100.0")

    projector = _projector_for_origin(
        enu_origin="lw1",
        enu_origin_lla=None,
        lw1_origin_lla=None,
    )

    assert projector is not None
    assert projector.origin_latitude_deg == pytest.approx(35.0)
    assert projector.origin_longitude_deg == pytest.approx(-78.0)
    assert projector.origin_altitude_m == pytest.approx(100.0)


def test_lw1_origin_can_be_supplied_by_config(tmp_path) -> None:
    origin_config = tmp_path / "origins.toml"
    origin_config.write_text(
        "\n".join(
            [
                "[origins.lw1]",
                "latitude_deg = 35.1",
                "longitude_deg = -78.2",
                "altitude_m = 123.4",
            ]
        ),
        encoding="utf-8",
    )

    projector = _projector_for_origin(
        enu_origin="lw1",
        enu_origin_lla=None,
        lw1_origin_lla=None,
        origin_config=origin_config,
    )

    assert projector is not None
    assert projector.origin_latitude_deg == pytest.approx(35.1)
    assert projector.origin_longitude_deg == pytest.approx(-78.2)
    assert projector.origin_altitude_m == pytest.approx(123.4)


def test_lw1_origin_rejects_example_placeholder(tmp_path) -> None:
    origin_config = tmp_path / "origins.toml"
    origin_config.write_text(
        "[origins.lw1]\nlatitude_deg = 0.0\nlongitude_deg = 0.0\naltitude_m = 0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="0,0,0"):
        _projector_for_origin(
            enu_origin="lw1",
            enu_origin_lla=None,
            lw1_origin_lla=None,
            origin_config=origin_config,
        )


def test_count_mismatch_action_fail_raises() -> None:
    table = pd.DataFrame({"method": ["RF raw"], "selected_count": [999]})
    count_audit = build_count_audit(table)

    with pytest.raises(RuntimeError, match="RF raw"):
        _handle_count_mismatch(count_audit, flight_name="example", action="fail")
