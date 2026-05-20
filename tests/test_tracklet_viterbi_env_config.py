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
