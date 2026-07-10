from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad.candidate_mixture_map_multistart import (
    CandidateMixtureMultiStartConfig,
)
from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_risk_reservoir_multistart import (
    CandidateRiskScoreConfig,
    main as pipeline_main,
    run_risk_reservoir_multistart,
)


def _candidate_rows() -> pd.DataFrame:
    records = []
    for time_s in range(4):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"low-risk-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "candidate_class_calibrated_score": 0.80,
                    "ranker_score": 0.80,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"high-risk-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 20),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "candidate_class_calibrated_score": 0.90,
                    "ranker_score": 0.90,
                    "predicted_sigma_m": 20.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_risk_reservoir_multistart_keeps_low_risk_candidate() -> None:
    result = run_risk_reservoir_multistart(
        _candidate_rows(),
        risk_config=CandidateRiskScoreConfig(uncertainty_weight=1.0),
        reservoir_config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=0,
            per_branch_top_n=0,
            max_candidates_per_frame=1,
            score_column="candidate_risk_adjusted_score",
        ),
        mixture_config=core.CandidateMixtureMapConfig(
            top_k=1,
            score_column="candidate_class_calibrated_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=100.0,
            iterations=3,
        ),
        multistart_config=CandidateMixtureMultiStartConfig(max_branch_starts=4),
        truth=_truth_rows(),
        oracle_top_k_values=(1,),
        max_truth_time_delta_s=0.1,
    )

    assert set(result.reservoir["track_id"]) == {
        "low-risk-0",
        "low-risk-1",
        "low-risk-2",
        "low-risk-3",
    }
    assert result.summary["mixture_config"]["top_k"] == 0
    assert result.summary["mixture_uses_risk_adjusted_score"] is False
    assert result.summary["truth_used_for_selection"] is False
    pooled = result.multistart.selected_result.summary["metrics"]["pooled"]
    assert pooled["rmse_3d_m"] < 0.01
    assert result.oracle_summary.loc[0, "oracle_top1_3d_m_mse"] == 0.0


def test_risk_reservoir_multistart_cli_writes_pipeline_artifacts(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = pipeline_main(
        [
            "--candidate-csv",
            f"pool={candidates_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--global-top-n",
            "1",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "1",
            "--smoothness-weight",
            "100",
            "--iterations",
            "3",
            "--oracle-top-k",
            "1",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    expected = [
        "mmuad_risk_reservoir_multistart_scored_candidates.csv",
        "mmuad_risk_reservoir_multistart_reservoir.csv",
        "mmuad_risk_reservoir_multistart_oracle_frames.csv",
        "mmuad_risk_reservoir_multistart_oracle_summary.csv",
        "mmuad_risk_reservoir_multistart_oracle_by_sequence.csv",
        "mmuad_risk_reservoir_multistart_summary.json",
        "mmuad_candidate_mixture_estimates.csv",
        "mmuad_candidate_mixture_assignments.csv",
        "mmuad_candidate_mixture_multistart_summary.csv",
    ]
    for filename in expected:
        assert (output_dir / filename).exists(), filename
    payload = json.loads(
        (output_dir / "mmuad_risk_reservoir_multistart_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["truth_used_for_selection"] is False
    assert payload["mixture_config"]["top_k"] == 0
    assert payload["reservoir_oracle"]["pooled"]["oracle_all_3d_m_mse"] == 0.0
