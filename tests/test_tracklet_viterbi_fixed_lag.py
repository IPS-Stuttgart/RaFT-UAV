import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import _events
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_fixed_lag import (
    run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay,
    select_fixed_lag_tracklet_viterbi_path,
)


def _rf_measurement(time_s: float, east_m: float, north_m: float = 0.0) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, north_m]),
        covariance=np.diag([4.0, 4.0]),
        source="rf",
    )


def _row(frame_index: int, track_id: int, time_s: float, east_m: float, cat_prob_uav: float) -> dict[str, float | int]:
    return {
        "frame_index": frame_index,
        "track_id": track_id,
        "time_s": time_s,
        "east_m": east_m,
        "north_m": 0.0,
        "up_m": 0.0,
        "cat_prob_uav": cat_prob_uav,
    }


def test_fixed_lag_tracklet_viterbi_uses_bounded_future_window():
    radar = pd.DataFrame(
        [
            _row(0, 1, 0.0, 0.0, 0.8),
            _row(0, 2, 0.0, 100.0, 0.99),
            _row(1, 1, 1.0, 10.0, 0.8),
            _row(1, 2, 1.0, 1000.0, 0.99),
        ]
    )

    _, _, viterbi_selected = (
        run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
            rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(1.0, 10.0)],
            radar=radar,
            lag_s=1.0,
            candidate_catprob_threshold=None,
            config=TrackletViterbiAssociationConfig(
                anchor_nis_weight=2.0,
                track_switch_cost=20.0,
                max_speed_penalty=10_000.0,
            ),
        )
    )

    assert viterbi_selected["track_id"].tolist() == [1, 1]
    assert viterbi_selected["association_lag_s"].tolist() == [1.0, 1.0]


def test_fixed_lag_tracklet_viterbi_conditions_on_previous_committed_choice():
    radar = pd.DataFrame(
        [
            _row(0, 1, 0.0, 0.0, 0.9),
            _row(0, 2, 0.0, 100.0, 0.01),
            _row(1, 1, 1.0, 10.0, 0.3),
            _row(1, 2, 1.0, 110.0, 0.99),
        ]
    )

    selected = select_fixed_lag_tracklet_viterbi_path(
        events=_events([], radar),
        anchors={},
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            catprob_weight=2.0,
            track_switch_cost=1_000.0,
            max_speed_mps=200.0,
            max_speed_penalty=0.0,
            transition_nis_weight=0.0,
            velocity_nis_weight=0.0,
            anchor_nis_weight=0.0,
        ),
        lag_s=1.0,
    )

    assert [int(row["track_id"]) for row in selected] == [1, 1]
    assert "association_prefix_constrained" not in selected[0]
    assert bool(selected[1]["association_prefix_constrained"])
    assert int(selected[1]["association_prefix_track_id"]) == 1
    assert float(selected[1]["association_prefix_time_s"]) == 0.0


def test_fixed_lag_replay_applies_do_no_harm_policy(monkeypatch):
    monkeypatch.setenv("RAFT_UAV_DO_NO_HARM_RADAR_UPDATES", "1")
    monkeypatch.setenv("RAFT_UAV_DNH_ANCHOR_SOFTEN_NIS", "0.1")
    monkeypatch.setenv("RAFT_UAV_DNH_ANCHOR_SKIP_NIS", "0.2")
    radar = pd.DataFrame(
        [
            _row(
                frame_index=0,
                track_id=99,
                time_s=10.0,
                east_m=1000.0,
                cat_prob_uav=0.99,
            ),
        ]
    )
    rf = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]),
        covariance=np.diag([1.0, 1.0, 1.0, 0.25, 0.25, 0.25]),
        source="rf",
    )

    records, accepted, replayed = (
        run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
            rf_measurements=[rf],
            radar=radar,
            lag_s=20.0,
            candidate_catprob_threshold=None,
            config=TrackletViterbiAssociationConfig(
                missed_detection_cost=1_000_000.0,
                anchor_nis_weight=0.0,
                range_gate_m=None,
            ),
        )
    )

    assert accepted.empty
    assert records[-1]["source"] == "radar"
    assert records[-1]["association_mode"] == "tracklet-viterbi-fixed-lag"
    assert records[-1]["update_action"] == "do_no_harm_skip"
    assert records[-1]["accepted"] is False
    assert records[-1]["time_s"] == 10.0
    np.testing.assert_allclose(records[-1]["state"][0], 100.0)
    assert replayed["association_replay_accepted"].tolist() == [False]
    assert replayed["association_replay_update_action"].tolist() == ["do_no_harm_skip"]
    assert replayed["association_update_policy"].tolist() == ["skip"]
