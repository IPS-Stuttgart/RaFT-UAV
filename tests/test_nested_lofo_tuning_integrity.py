from __future__ import annotations

import numpy as np
import pytest

from raft_uav.experiments.nested_lofo_tuning import _select_candidate, main


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
