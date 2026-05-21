from __future__ import annotations

from raft_uav import best_non_oracle_cli


def _arg_after(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_best_non_oracle_defaults_to_imm_fixed_lag_tracklet_viterbi() -> None:
    args = best_non_oracle_cli._parse_args(["dataset", "--flight", "Opt2"])

    forwarded = best_non_oracle_cli.build_tracklet_cli_argv(args)

    assert forwarded[:5] == [
        "--tracklet-variant",
        "range-covariance",
        "--tracklet-replay-tracker",
        "imm",
        "run-baseline",
    ]
    assert _arg_after(forwarded, "--radar-association") == "tracklet-viterbi"
    assert _arg_after(forwarded, "--smoother") == "fixed-lag"
    assert _arg_after(forwarded, "--smoother-lag-s") == "20"
    assert _arg_after(forwarded, "--robust-update") == "student-t"
    assert all("oracle" not in token for token in forwarded)


def test_best_non_oracle_main_forwards_expanded_command(monkeypatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_main(argv: list[str]) -> int:
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(best_non_oracle_cli._tracklet_cli, "main", fake_main)

    status = best_non_oracle_cli.main(
        [
            "dataset",
            "--flight",
            "Opt1",
            "--output-dir",
            "outputs/custom",
            "--smoother-lag-s",
            "12.5",
            "--robust-update",
            "huber",
        ]
    )

    assert status == 7
    assert _arg_after(seen["argv"], "--flight") == "Opt1"
    assert _arg_after(seen["argv"], "--output-dir") == "outputs/custom"
    assert _arg_after(seen["argv"], "--smoother-lag-s") == "12.5"
    assert _arg_after(seen["argv"], "--robust-update") == "huber"


def test_best_non_oracle_dry_run_does_not_call_tracker(monkeypatch, capsys) -> None:
    def fail_main(argv: list[str]) -> int:
        raise AssertionError(f"unexpected tracking call with {argv!r}")

    monkeypatch.setattr(best_non_oracle_cli._tracklet_cli, "main", fail_main)

    status = best_non_oracle_cli.main(["dataset", "--flight", "Opt3", "--dry-run"])

    assert status == 0
    printed = capsys.readouterr().out
    assert "raft-uav --tracklet-variant range-covariance" in printed
    assert "--tracklet-replay-tracker imm" in printed
    assert "--smoother fixed-lag" in printed
    assert "--flight Opt3" in printed


def test_best_non_oracle_forwards_calibration_learned_candidate_and_velocity() -> None:
    args = best_non_oracle_cli._parse_args(
        [
            "dataset",
            "--flight",
            "Opt2",
            "--calibration-bundle",
            "outputs/lofo_calibration/Opt2/calibration_bundle.json",
            "--learned-candidate-model",
            "outputs/lofo_calibration/Opt2/radar_association_model.json",
            "--learned-candidate-score-mode",
            "additive",
            "--enable-radar-velocity-update",
            "--radar-velocity-std-mps",
            "20",
        ]
    )

    forwarded = best_non_oracle_cli.build_tracklet_cli_argv(args)

    assert _arg_after(forwarded, "--calibration-bundle") == (
        "outputs/lofo_calibration/Opt2/calibration_bundle.json"
    )
    assert _arg_after(forwarded, "--tracklet-learned-candidate-model") == (
        "outputs/lofo_calibration/Opt2/radar_association_model.json"
    )
    assert _arg_after(forwarded, "--tracklet-learned-candidate-score-mode") == "additive"
    assert "--enable-radar-velocity-update" in forwarded
    assert _arg_after(forwarded, "--radar-velocity-std-mps") == "20"
