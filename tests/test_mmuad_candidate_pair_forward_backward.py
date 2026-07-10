from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
    main as pair_forward_backward_main,
    pair_forward_backward_summary,
)


def _constant_velocity_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "source": ["lidar", "lidar", "radar", "lidar"],
            "track_id": ["start", "smooth", "kink", "finish"],
            "candidate_branch": ["raw", "raw", "translated", "raw"],
            "x_m": [0.0, 1.0, 5.0, 2.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "ranker_score": [0.5, 0.2, 0.9, 0.5],
            "predicted_sigma_m": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _config() -> CandidatePairForwardBackwardConfig:
    return CandidatePairForwardBackwardConfig(
        score_column="ranker_score",
        fallback_score_columns=("confidence",),
        sigma_column="predicted_sigma_m",
        score_normalization="minmax",
        score_weight=1.0,
        sigma_log_weight=0.0,
        transition_distance_std_m=100.0,
        transition_speed_std_mps=100.0,
        max_speed_mps=1_000.0,
        speed_gate_penalty=0.0,
        acceleration_std_mps2=0.5,
        max_acceleration_mps2=1_000.0,
        acceleration_gate_penalty=0.0,
        source_switch_penalty=0.0,
        branch_switch_penalty=0.0,
        track_continuation_bonus=0.0,
    )


def test_pair_forward_backward_prefers_constant_velocity_candidate() -> None:
    augmented = attach_pair_forward_backward_candidate_prior(
        _constant_velocity_candidates(),
        config=_config(),
    ).rows

    middle = augmented.loc[augmented["time_s"] == 1.0].sort_values(
        "candidate_pair_forward_backward_score",
        ascending=False,
    )

    assert middle.iloc[0]["track_id"] == "smooth"
    assert (
        middle.iloc[0]["candidate_pair_forward_backward_score"]
        > middle.iloc[1]["candidate_pair_forward_backward_score"]
    )
    assert middle.iloc[0]["candidate_pair_forward_backward_min_acceleration_mps2"] == 0.0
    assert middle.iloc[1]["candidate_pair_forward_backward_min_acceleration_mps2"] > 1.0


def test_pair_forward_backward_posteriors_sum_to_one_per_frame() -> None:
    augmented = attach_pair_forward_backward_candidate_prior(
        _constant_velocity_candidates(),
        config=_config(),
    ).rows

    sums = augmented.groupby(["sequence_id", "time_s"])[
        "candidate_pair_forward_backward_score"
    ].sum()

    assert np.allclose(sums.to_numpy(float), 1.0)
    summary = pair_forward_backward_summary(augmented)
    assert summary["posterior_sum_error_max"] < 1.0e-12
    assert summary["frame_count"] == 3


def test_pair_forward_backward_handles_single_frame_sequence() -> None:
    rows = _constant_velocity_candidates().loc[lambda frame: frame["time_s"] == 1.0]

    augmented = attach_pair_forward_backward_candidate_prior(rows, config=_config()).rows

    assert np.isclose(augmented["candidate_pair_forward_backward_score"].sum(), 1.0)
    assert (augmented["candidate_pair_forward_backward_pair_state_count"] == 0).all()


def test_pair_forward_backward_cli_writes_candidates_summary_and_mixture(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "pair_candidates.csv"
    summary_json = tmp_path / "pair_summary.json"
    mixture_dir = tmp_path / "mixture"
    _constant_velocity_candidates().to_csv(candidates_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    ).to_csv(truth_csv, index=False)

    status = pair_forward_backward_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--transition-distance-std-m",
            "100",
            "--transition-speed-std-mps",
            "100",
            "--max-speed-mps",
            "1000",
            "--speed-gate-penalty",
            "0",
            "--acceleration-std-mps2",
            "0.5",
            "--max-acceleration-mps2",
            "1000",
            "--acceleration-gate-penalty",
            "0",
            "--source-switch-penalty",
            "0",
            "--branch-switch-penalty",
            "0",
            "--track-continuation-bonus",
            "0",
            "--mixture-output-dir",
            str(mixture_dir),
            "--mixture-truth-csv",
            str(truth_csv),
            "--mixture-top-k",
            "0",
            "--mixture-smoothness-weight",
            "10",
            "--mixture-iterations",
            "2",
        ]
    )

    assert status == 0
    assert output_csv.exists()
    assert summary_json.exists()
    assert (mixture_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["truth_used_for_candidate_prior"] is False
    assert payload["summary"]["posterior_sum_error_max"] < 1.0e-12
    assert payload["mixture_summary"]["estimate_rows"] == 3
