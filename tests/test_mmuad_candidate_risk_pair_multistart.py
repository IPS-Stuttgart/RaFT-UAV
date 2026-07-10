from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad.candidate_mixture_map_multistart import CandidateMixtureMultiStartConfig
from raft_uav.mmuad.candidate_pair_forward_backward import CandidatePairForwardBackwardConfig
from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_risk_pair_multistart import (
    main as risk_pair_main,
    run_risk_pair_multistart,
)
from raft_uav.mmuad.candidate_risk_reservoir_multistart import CandidateRiskScoreConfig


def _candidates() -> pd.DataFrame:
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
            "candidate_class_calibrated_score": [0.5, 0.2, 0.9, 0.5],
            "ranker_score": [0.5, 0.2, 0.9, 0.5],
            "predicted_sigma_m": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def _pair_config() -> CandidatePairForwardBackwardConfig:
    return CandidatePairForwardBackwardConfig(
        score_column="candidate_class_calibrated_score",
        fallback_score_columns=("candidate_risk_adjusted_score", "ranker_score"),
        sigma_column="predicted_sigma_m",
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


def test_risk_pair_multistart_preserves_pool_and_prefers_smooth_middle() -> None:
    result = run_risk_pair_multistart(
        _candidates(),
        risk_config=CandidateRiskScoreConfig(
            score_column="candidate_class_calibrated_score",
            fallback_score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            uncertainty_weight=0.0,
        ),
        reservoir_config=ReservoirConfig(
            global_top_n=2,
            per_source_top_n=0,
            per_branch_top_n=0,
            max_candidates_per_frame=2,
            score_column="candidate_risk_adjusted_score",
        ),
        pair_config=_pair_config(),
        mixture_config=core.CandidateMixtureMapConfig(
            top_k=0,
            score_column="candidate_pair_forward_backward_score",
            fallback_score_columns=("candidate_class_calibrated_score",),
            sigma_column="predicted_sigma_m",
            score_normalization="none",
            score_weight=4.0,
            sigma_log_weight=0.0,
            smoothness_weight=10.0,
            iterations=3,
        ),
        multistart_config=CandidateMixtureMultiStartConfig(
            include_score_top1=True,
            include_frame_median=True,
            include_branch_starts=True,
        ),
        truth=_truth(),
    )

    middle = result.pair_candidates.loc[result.pair_candidates["time_s"] == 1.0].sort_values(
        "candidate_pair_forward_backward_score",
        ascending=False,
    )
    assert middle.iloc[0]["track_id"] == "smooth"
    sums = result.pair_candidates.groupby(["sequence_id", "time_s"])[
        "candidate_pair_forward_backward_score"
    ].sum()
    assert np.allclose(sums.to_numpy(float), 1.0)
    assert len(result.reservoir) == 4
    assert result.summary["truth_used_for_selection"] is False
    assert result.summary["mixture_uses_pair_posterior"] is True
    assert result.summary["pair_prior"]["posterior_sum_error_max"] < 1.0e-12
    assert result.oracle_summary.loc[0, "oracle_all_3d_m_mse"] == 0.0


def test_risk_pair_multistart_cli_writes_intermediate_and_selected_outputs(
    tmp_path: Path,
) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _truth().to_csv(truth_csv, index=False)

    status = risk_pair_main(
        [
            "--candidate-csv",
            f"raw={candidates_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--global-top-n",
            "2",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "2",
            "--uncertainty-weight",
            "0",
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
            "--smoothness-weight",
            "10",
            "--iterations",
            "2",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_risk_pair_multistart_scored_candidates.csv").exists()
    assert (output_dir / "mmuad_risk_pair_multistart_reservoir.csv").exists()
    assert (output_dir / "mmuad_risk_pair_multistart_pair_candidates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_multistart_summary.csv").exists()
    assert (output_dir / "mmuad_risk_pair_multistart_oracle_summary.csv").exists()
    payload = json.loads(
        (output_dir / "mmuad_risk_pair_multistart_summary.json").read_text(encoding="utf-8")
    )
    assert payload["truth_used_for_selection"] is False
    assert payload["pair_prior"]["posterior_sum_error_max"] < 1.0e-12
