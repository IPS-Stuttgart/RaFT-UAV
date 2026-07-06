from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_forward_backward import (
    CandidateForwardBackwardConfig,
    attach_forward_backward_candidate_prior,
    forward_backward_summary,
    main as forward_backward_main,
)


def _outlier_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001", "seq001"],
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "source": ["lidar", "lidar", "radar", "lidar"],
            "track_id": ["smooth", "smooth", "outlier", "smooth"],
            "candidate_branch": ["raw", "raw", "raw", "raw"],
            "x_m": [0.0, 1.0, 100.0, 2.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "ranker_score": [0.6, 0.4, 0.99, 0.6],
            "predicted_sigma_m": [2.0, 2.0, 2.0, 2.0],
            "confidence": [0.6, 0.4, 0.99, 0.6],
        }
    )


def test_forward_backward_prior_rejects_temporal_outlier() -> None:
    augmented = attach_forward_backward_candidate_prior(
        _outlier_candidates(),
        config=CandidateForwardBackwardConfig(
            score_column="ranker_score",
            transition_distance_std_m=1.0,
            transition_speed_std_mps=2.0,
            max_speed_mps=20.0,
            speed_gate_penalty=50.0,
            source_switch_penalty=0.0,
            branch_switch_penalty=0.0,
            track_continuation_bonus=0.0,
        ),
    ).rows

    middle = augmented.loc[augmented["time_s"] == 1.0].set_index("track_id")
    assert middle.loc["smooth", "candidate_forward_backward_score"] > 0.99
    assert middle.loc["outlier", "candidate_forward_backward_score"] < 0.01
    assert middle.loc["smooth", "candidate_forward_backward_rank"] == 1.0


def test_forward_backward_source_switch_penalty_prefers_continuity() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq001"] * 4,
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "source": ["lidar", "lidar", "radar", "lidar"],
            "track_id": ["a", "a", "b", "a"],
            "candidate_branch": ["raw"] * 4,
            "x_m": [0.0, 1.0, 1.0, 2.0],
            "y_m": [0.0] * 4,
            "z_m": [0.0] * 4,
            "ranker_score": [0.5, 0.45, 0.55, 0.5],
            "predicted_sigma_m": [2.0] * 4,
            "confidence": [0.5, 0.45, 0.55, 0.5],
        }
    )

    augmented = attach_forward_backward_candidate_prior(
        candidates,
        config=CandidateForwardBackwardConfig(
            score_column="ranker_score",
            transition_distance_std_m=2.0,
            transition_speed_std_mps=10.0,
            source_switch_penalty=2.0,
            track_continuation_bonus=0.0,
        ),
    ).rows

    middle = augmented.loc[augmented["time_s"] == 1.0].set_index("source")
    assert middle.loc["lidar", "candidate_forward_backward_score"] > middle.loc[
        "radar", "candidate_forward_backward_score"
    ]


def test_forward_backward_probabilities_sum_to_one() -> None:
    augmented = attach_forward_backward_candidate_prior(_outlier_candidates()).rows

    sums = augmented.groupby(["sequence_id", "time_s"])[
        "candidate_forward_backward_score"
    ].sum()
    assert sums.tolist() == pytest.approx([1.0, 1.0, 1.0])
    summary = forward_backward_summary(augmented)
    assert summary["frame_count"] == 3
    assert summary["posterior_sum_error_max"] == pytest.approx(0.0, abs=1.0e-12)


def test_forward_backward_summary_handles_missing_score_column() -> None:
    summary = forward_backward_summary(
        _outlier_candidates(),
        score_column="missing_forward_backward_score",
    )

    assert summary["row_count"] == 4
    assert summary["frame_count"] == 3
    assert summary["score_column"] == "missing_forward_backward_score"
    assert summary["posterior_sum_error_max"] == pytest.approx(0.0)
    assert summary["top_posterior_mean"] is None
    assert summary["top_candidate_source_counts"] == {}
    assert summary["top_candidate_branch_counts"] == {}


def test_forward_backward_cli_writes_prior_and_mixture_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "forward_backward_candidates.csv"
    summary_json = tmp_path / "forward_backward_summary.json"
    mixture_dir = tmp_path / "mixture"
    _outlier_candidates().to_csv(candidates_csv, index=False)

    status = forward_backward_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--score-column",
            "ranker_score",
            "--transition-distance-std-m",
            "1",
            "--transition-speed-std-mps",
            "2",
            "--max-speed-mps",
            "20",
            "--speed-gate-penalty",
            "50",
            "--mixture-output-dir",
            str(mixture_dir),
            "--mixture-top-k",
            "3",
            "--mixture-smoothness-weight",
            "10",
        ]
    )

    assert status == 0
    assert output_csv.exists()
    assert summary_json.exists()
    assert (mixture_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (mixture_dir / "mmuad_candidate_mixture_assignments.csv").exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["summary"]["frame_count"] == 3
    assert payload["mixture_output_dir"] == str(mixture_dir)
