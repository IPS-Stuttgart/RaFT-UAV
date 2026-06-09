from __future__ import annotations

from raft_uav import tracklet_viterbi_cli


def test_tracklet_viterbi_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv(tracklet_viterbi_cli._MAX_CANDIDATES_PER_FRAME_ENV, "11")
    monkeypatch.setenv(tracklet_viterbi_cli._MAX_CANDIDATE_POOL_ENV, "25")
    monkeypatch.setenv(tracklet_viterbi_cli._MAX_CANDIDATES_PER_TRACK_ENV, "2")
    monkeypatch.setenv(tracklet_viterbi_cli._BELOW_CATPROB_PENALTY_ENV, "1.5")
    monkeypatch.setenv(tracklet_viterbi_cli._TRACK_SUPPORT_WEIGHT_ENV, "0.25")
    monkeypatch.setenv(tracklet_viterbi_cli._MAX_TRACK_SUPPORT_REWARD_ENV, "3.0")

    config = tracklet_viterbi_cli._tracklet_config_from_environment()

    assert config.max_candidates_per_frame == 11
    assert config.max_candidate_pool_per_frame == 25
    assert config.max_candidates_per_track_id == 2
    assert config.below_catprob_threshold_penalty == 1.5
    assert config.track_support_weight == 0.25
    assert config.max_track_support_reward == 3.0


def test_tracklet_viterbi_config_reads_core_runtime_environment(monkeypatch) -> None:
    monkeypatch.setenv(tracklet_viterbi_cli._TRACK_SWITCH_COST_ENV, "17")
    monkeypatch.setenv(tracklet_viterbi_cli._ANCHOR_NIS_WEIGHT_ENV, "0.7")
    monkeypatch.setenv(tracklet_viterbi_cli._TRANSITION_POSITION_STD_M_ENV, "55")
    monkeypatch.setenv(tracklet_viterbi_cli._TRANSITION_SPEED_STD_MPS_ENV, "22")
    monkeypatch.setenv(tracklet_viterbi_cli._VELOCITY_STD_MPS_ENV, "14")
    monkeypatch.setenv(tracklet_viterbi_cli._MAX_SPEED_MPS_ENV, "65")
    monkeypatch.setenv(tracklet_viterbi_cli._RANGE_GATE_M_ENV, "0")
    monkeypatch.setenv(tracklet_viterbi_cli._RANGE_GATE_SLACK_M_ENV, "200")
    monkeypatch.setenv(tracklet_viterbi_cli._USE_RF_ANCHOR_ENV, "0")

    config = tracklet_viterbi_cli._tracklet_config_from_environment()

    assert config.track_switch_cost == 17.0
    assert config.anchor_nis_weight == 0.7
    assert config.transition_position_std_m == 55.0
    assert config.transition_speed_std_mps == 22.0
    assert config.velocity_std_mps == 14.0
    assert config.max_speed_mps == 65.0
    assert config.range_gate_m is None
    assert config.range_gate_slack_m == 200.0
    assert config.use_rf_anchor is False


def test_tracklet_viterbi_config_supports_legacy_candidate_env_alias(monkeypatch) -> None:
    monkeypatch.delenv(tracklet_viterbi_cli._MAX_CANDIDATES_PER_FRAME_ENV, raising=False)
    monkeypatch.setenv(tracklet_viterbi_cli._MAX_CANDIDATES_PER_FRAME_LEGACY_ENV, "13")

    config = tracklet_viterbi_cli._tracklet_config_from_environment()

    assert config.max_candidates_per_frame == 13


def test_tracklet_viterbi_canonical_candidate_env_overrides_legacy_alias(monkeypatch) -> None:
    monkeypatch.setenv(tracklet_viterbi_cli._MAX_CANDIDATES_PER_FRAME_LEGACY_ENV, "13")
    monkeypatch.setenv(tracklet_viterbi_cli._MAX_CANDIDATES_PER_FRAME_ENV, "17")

    config = tracklet_viterbi_cli._tracklet_config_from_environment()

    assert config.max_candidates_per_frame == 17


def test_tracklet_cli_leaves_runtime_tracklet_args_for_runtime_parser() -> None:
    remaining, updates = tracklet_viterbi_cli._extract_tracklet_args(
        [
            "run-baseline",
            "/data/aerpaw",
            "--tracklet-track-support-weight",
            "0.9",
            "--tracklet-max-candidates-per-frame",
            "12",
            "--tracklet-variant",
            "retention",
        ]
    )

    assert updates == {tracklet_viterbi_cli._TRACKLET_VARIANT_ENV: "retention"}
    assert remaining == [
        "run-baseline",
        "/data/aerpaw",
        "--tracklet-track-support-weight",
        "0.9",
        "--tracklet-max-candidates-per-frame",
        "12",
    ]
