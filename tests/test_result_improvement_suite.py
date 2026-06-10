from pathlib import Path

from raft_uav.research.result_improvement_suite import (
    CommandSpec,
    DEFAULT_RUNTIME_ENV,
    ImprovementSuiteConfig,
    build_improvement_suite_plan,
)


def test_command_shell_line_includes_environment_prefix() -> None:
    spec = CommandSpec("demo", ("python", "script.py", "a b"), {"B": "two words", "A": "1"})

    line = spec.shell_line()

    assert line.startswith("A=1 B='two words'")
    assert "python script.py 'a b'" in line


def test_suite_plan_contains_all_major_workflow_steps() -> None:
    cfg = ImprovementSuiteConfig(
        dataset_root=Path("data"),
        flights=("Opt1",),
        methods=("imm_tracklet_viterbi_fixed_lag",),
    )

    names = [spec.name for spec in build_improvement_suite_plan(cfg)]

    assert "lofo_time_offset_calibration" in names
    assert "lofo_radar_covariance_tuning" in names
    assert "nested_lofo_tuning" in names
    assert "leave_flight_out_sota" in names
    assert "oracle_gap_imm_tracklet_viterbi_fixed_lag_Opt1" in names
    assert "constrained_leaderboard_ranking" in names


def test_suite_plan_can_disable_diagnostic_steps() -> None:
    cfg = ImprovementSuiteConfig(
        dataset_root=Path("data"),
        include_oracle_gap=False,
        include_constrained_ranking=False,
    )

    names = [spec.name for spec in build_improvement_suite_plan(cfg)]

    assert not any(name.startswith("oracle_gap_") for name in names)
    assert "constrained_leaderboard_ranking" not in names


def test_suite_uses_tracklet_cli_candidate_env_key() -> None:
    assert DEFAULT_RUNTIME_ENV["RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_FRAME"] == "12"
    assert "RAFT_UAV_TRACKLET_MAX_CANDIDATES" not in DEFAULT_RUNTIME_ENV


def test_suite_enables_do_no_harm_policy_gate_in_default_runtime_env() -> None:
    assert DEFAULT_RUNTIME_ENV["RAFT_UAV_DO_NO_HARM_RADAR_UPDATES"] == "1"
    assert DEFAULT_RUNTIME_ENV["RAFT_UAV_DO_NO_HARM_RADAR_UPDATE_POLICY"] == "1"


def test_sota_command_carries_fixed_runtime_env() -> None:
    config = ImprovementSuiteConfig(
        dataset_root=Path("data/raw/AADM2025Dryad"),
        output_dir=Path("outputs/result_improvement_suite"),
        flights=("Opt1", "Opt2"),
        methods=("imm_tracklet_viterbi_fixed_lag",),
        include_time_offset_calibration=False,
        include_covariance_tuning=False,
        include_nested_tuning=False,
        include_oracle_gap=False,
        include_constrained_ranking=False,
    )

    commands = build_improvement_suite_plan(config)

    assert len(commands) == 1
    assert commands[0].name == "leave_flight_out_sota"
    assert commands[0].env["RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_FRAME"] == "12"
    assert commands[0].env["RAFT_UAV_DO_NO_HARM_RADAR_UPDATES"] == "1"
    assert "RAFT_UAV_TRACKLET_MAX_CANDIDATES" not in commands[0].env
