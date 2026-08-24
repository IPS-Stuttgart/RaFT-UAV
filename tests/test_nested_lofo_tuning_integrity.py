from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from raft_uav.experiments.nested_lofo_tuning import Candidate, _run_candidate, _select_candidate, main


def test_main_rejects_duplicate_flights(tmp_path) -> None:
    with pytest.raises(ValueError, match="flight values must be unique"):
        main(
            [
                str(tmp_path),
                "--flight",
                "Opt1",
                "--flight",
                "Opt2",
                "--flight",
                "Opt1",
                "--candidate",
                "base=",
                "--dry-run",
            ]
        )


def test_main_rejects_duplicate_candidate_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="candidate name values must be unique"):
        main(
            [
                str(tmp_path),
                "--flight",
                "Opt1",
                "--flight",
                "Opt2",
                "--candidate",
                "base=--radar-association catprob",
                "--candidate",
                "base=--radar-association prediction-nis",
                "--dry-run",
            ]
        )


def test_run_candidate_preserves_whitespace_in_command_values(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset root"
    output_root = tmp_path / "output root"
    args = SimpleNamespace(
        dataset_root=dataset_root,
        output_dir=output_root,
        skip_existing=False,
        dry_run=False,
        base_command=(
            "{python} -m raft_uav.cli run-baseline {dataset_root} "
            "--flight {flight} --output-dir {output_dir}"
        ),
    )
    captured: dict[str, object] = {}

    def fake_run(command, *, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr("raft_uav.experiments.nested_lofo_tuning.subprocess.run", fake_run)

    candidate = Candidate(name="base", args=("--radar-association", "catprob"))
    _run_candidate(args, candidate, "Opt 1", split="holdout Opt 2/train")

    command = captured["command"]
    assert isinstance(command, list)
    assert captured["check"] is True
    assert command[command.index("run-baseline") + 1] == str(dataset_root)
    assert command[command.index("--flight") + 1] == "Opt 1"
    assert command[command.index("--output-dir") + 1] == str(output_root / "holdout Opt 2/train" / "base")
    assert command[-2:] == ["--radar-association", "catprob"]


def test_select_candidate_requires_every_training_flight() -> None:
    rows = [
        {"candidate": "complete", "flight": "Opt1", "metric_value": 10.0},
        {"candidate": "complete", "flight": "Opt2", "metric_value": 10.0},
        {"candidate": "partial", "flight": "Opt1", "metric_value": 1.0},
        {"candidate": "partial", "flight": "Opt2", "metric_value": np.nan},
    ]

    selected = _select_candidate(
        rows,
        aggregate="mean",
        expected_flights=["Opt1", "Opt2"],
    )

    assert selected == {"candidate": "complete", "metric_value": 10.0}


def test_select_candidate_rejects_duplicate_candidate_flight_rows() -> None:
    rows = [
        {"candidate": "ambiguous", "flight": "Opt1", "metric_value": 1.0},
        {"candidate": "ambiguous", "flight": "Opt1", "metric_value": 2.0},
        {"candidate": "ambiguous", "flight": "Opt2", "metric_value": 1.0},
        {"candidate": "complete", "flight": "Opt1", "metric_value": 3.0},
        {"candidate": "complete", "flight": "Opt2", "metric_value": 3.0},
    ]

    selected = _select_candidate(
        rows,
        aggregate="mean",
        expected_flights=["Opt1", "Opt2"],
    )

    assert selected == {"candidate": "complete", "metric_value": 3.0}
