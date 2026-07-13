from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig
from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_reservoir_mixture_map_grouped import (
    main as grouped_reservoir_main,
    run_grouped_reservoir_mixture_map,
)


def _duplicate_hypotheses() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "cluster-1@raw",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 1,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "cluster-1@calibrated",
                "candidate_branch": "source_translation_calibrated",
                "mmuad_calibration_origin_row": 1,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "cluster-2@raw",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 2,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )


def _reservoir_config() -> ReservoirConfig:
    return ReservoirConfig(
        global_top_n=3,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=3,
        score_column="ranker_score",
    )


def _neutral_mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=1,
        score_column="candidate_reservoir_score",
        score_normalization="none",
        score_weight=0.0,
        sigma_log_weight=0.0,
        loss="squared",
        smoothness_weight=0.0,
        iterations=1,
    )


def test_grouped_reservoir_removes_duplicate_representation_mass() -> None:
    result = run_grouped_reservoir_mixture_map(
        _duplicate_hypotheses(),
        reservoir_config=_reservoir_config(),
        mixture_config=_neutral_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
    )

    assignments = result.grouped_result.mixture_result.assignments
    group_mass = assignments.groupby(
        "mixture_hypothesis_group"
    )["mixture_final_weight"].sum()

    assert len(result.reservoir) == 3
    assert sorted(group_mass.to_numpy(float)) == pytest.approx([0.5, 0.5])
    assert result.summary["mixture_config"]["top_k"] == 0
    assert result.summary["hypothesis_grouping"][
        "duplicate_hypothesis_group_count"
    ] == 1


def test_grouped_reservoir_cli_writes_composed_diagnostics(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    output_dir = tmp_path / "out"
    _duplicate_hypotheses().to_csv(candidates, index=False)

    status = grouped_reservoir_main(
        [
            "--candidate-csv",
            f"union={candidates}",
            "--output-dir",
            str(output_dir),
            "--global-top-n",
            "3",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "3",
            "--reservoir-score-column",
            "ranker_score",
            "--mixture-score-column",
            "candidate_reservoir_score",
            "--score-normalization",
            "none",
            "--score-weight",
            "0",
            "--sigma-log-weight",
            "0",
            "--loss",
            "squared",
            "--smoothness-weight",
            "0",
            "--iterations",
            "1",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_grouped_reservoir_candidates.csv").exists()
    assert (output_dir / "mmuad_group_corrected_candidates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_assignments.csv").exists()
    summary = json.loads(
        (output_dir / "mmuad_grouped_reservoir_mixture_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["reservoir"]["reservoir_candidate_rows"] == 3
    assert summary["hypothesis_grouping"]["duplicate_hypothesis_group_count"] == 1
    assert summary["mixture_config"]["top_k"] == 0
