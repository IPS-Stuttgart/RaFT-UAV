import pandas as pd

from raft_uav import tracklet_viterbi_fixed_lag_cli as fixed_lag_cli


def test_fixed_lag_cli_forwards_explicit_gate_thresholds(monkeypatch):
    captured: dict[str, object] = {}

    def fake_runner(**kwargs: object):
        captured.update(kwargs)
        return [], pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        fixed_lag_cli,
        "run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay",
        fake_runner,
    )
    gate_thresholds = {"rf": 3.0, "radar": 4.0}
    safety_thresholds = {"rf": 30.0, "radar": 40.0}

    fixed_lag_cli.run_async_cv_baseline_with_radar_association(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="tracklet-viterbi-fixed-lag",
        gate_thresholds_by_source=gate_thresholds,
        safety_gate_thresholds_by_source=safety_thresholds,
    )

    assert captured["gate_thresholds_by_source"] == gate_thresholds
    assert captured["safety_gate_thresholds_by_source"] == safety_thresholds
