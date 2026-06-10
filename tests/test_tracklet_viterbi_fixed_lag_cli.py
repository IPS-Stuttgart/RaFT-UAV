import pandas as pd

from raft_uav import tracklet_viterbi_fixed_lag_cli as fixed_lag_cli


def test_fixed_lag_wrapper_accepts_current_base_cli_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fixed_lag_runner(**kwargs: object):
        captured.update(kwargs)
        return [], pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        fixed_lag_cli,
        "run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay",
        fake_fixed_lag_runner,
    )

    records, selected = fixed_lag_cli.run_async_cv_baseline_with_radar_association(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="tracklet-viterbi-fixed-lag",
        radar_covariance_model="geometry",
        radar_range_std_m=9.0,
        radar_range_std_fraction=0.01,
        radar_crossrange_angle_std_deg=2.0,
        radar_crossrange_min_std_m=4.0,
        radar_crossrange_max_std_m=90.0,
        paper_compatible_catprob_threshold=0.3,
        paper_compatible_bootstrap_source="first-event",
        paper_compatible_empirical_covariance=False,
    )

    assert records == []
    assert selected.empty
    assert captured["lag_s"] == 20.0
    assert "radar_covariance_model" not in captured
    assert "paper_compatible_catprob_threshold" not in captured


def test_fixed_lag_wrapper_forwards_current_base_cli_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    def fake_base_runner(**kwargs: object):
        captured.update(kwargs)
        return [], pd.DataFrame()

    monkeypatch.setattr(fixed_lag_cli, "_base_radar_association_runner", fake_base_runner)

    records, selected = fixed_lag_cli.run_async_cv_baseline_with_radar_association(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="prediction-nis",
        radar_covariance_model="geometry",
        radar_range_std_m=9.0,
        radar_range_std_fraction=0.01,
        radar_crossrange_angle_std_deg=2.0,
        radar_crossrange_min_std_m=4.0,
        radar_crossrange_max_std_m=90.0,
        paper_compatible_catprob_threshold=0.3,
        paper_compatible_bootstrap_source="first-event",
        paper_compatible_empirical_covariance=False,
    )

    assert records == []
    assert selected.empty
    assert captured["association"] == "prediction-nis"
    assert captured["radar_covariance_model"] == "geometry"
    assert captured["radar_range_std_m"] == 9.0
    assert captured["radar_range_std_fraction"] == 0.01
    assert captured["radar_crossrange_angle_std_deg"] == 2.0
    assert captured["radar_crossrange_min_std_m"] == 4.0
    assert captured["radar_crossrange_max_std_m"] == 90.0
    assert captured["paper_compatible_catprob_threshold"] == 0.3
    assert captured["paper_compatible_bootstrap_source"] == "first-event"
    assert captured["paper_compatible_empirical_covariance"] is False
