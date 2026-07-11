from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_spatial_topk import (
    SpatialHypothesisGroupTopKConfig,
    main as spatial_group_topk_main,
    run_spatial_group_topk_candidate_mixture_map,
    select_spatial_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_topk import (
    HypothesisGroupTopKConfig,
    select_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def _candidate_rows() -> pd.DataFrame:
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
                "ranker_score": 0.99,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "cluster-1@calibrated",
                "candidate_branch": "source_translation_calibrated",
                "mmuad_calibration_origin_row": 1,
                "x_m": 0.05,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.98,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "cluster-2@raw",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 2,
                "x_m": 0.2,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.95,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "radar",
                "track_id": "cluster-3@raw",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 3,
                "x_m": 10.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.94,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "radar",
                "track_id": "cluster-4@raw",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 4,
                "x_m": 20.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.70,
                "predicted_sigma_m": 1.0,
            },
        ]
    )


def _mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=2,
        score_column="ranker_score",
        score_normalization="none",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
        loss="squared",
        smoothness_weight=0.0,
        iterations=1,
    )


def test_spatial_group_topk_recovers_a_distinct_lower_score_group() -> None:
    selected, summary = select_spatial_hypothesis_group_topk(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=SpatialHypothesisGroupTopKConfig(
            group_top_k=2,
            max_siblings_per_group=1,
            diversity_weight=1.0,
            diversity_scale_m=1.0,
            diversity_cap_m=30.0,
        ),
    )

    assert selected["mmuad_calibration_origin_row"].tolist() == [1, 3]
    assert selected["mixture_spatial_group_rank"].tolist() == [1, 2]
    assert selected["mixture_spatial_group_min_distance_m"].iloc[1] > 9.0
    assert summary["selected_groups_per_frame_mean"] == 2.0


def test_zero_diversity_weight_matches_score_only_group_topk() -> None:
    ordinary, _ = select_hypothesis_group_topk(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=HypothesisGroupTopKConfig(
            group_top_k=2,
            max_siblings_per_group=1,
        ),
    )
    spatial, _ = select_spatial_hypothesis_group_topk(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=SpatialHypothesisGroupTopKConfig(
            group_top_k=2,
            max_siblings_per_group=1,
            diversity_weight=0.0,
        ),
    )

    assert spatial["mmuad_calibration_origin_row"].tolist() == ordinary[
        "mmuad_calibration_origin_row"
    ].tolist()


def test_spatial_group_topk_disables_second_row_truncation() -> None:
    result = run_spatial_group_topk_candidate_mixture_map(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=SpatialHypothesisGroupTopKConfig(
            group_top_k=2,
            max_siblings_per_group=2,
            diversity_weight=1.0,
            diversity_scale_m=1.0,
        ),
    )

    selected_groups = set(result.selected_candidates["mmuad_calibration_origin_row"])
    assignment_groups = set(
        result.grouped_result.mixture_result.assignments[
            "mixture_hypothesis_group"
        ]
    )
    assert selected_groups == {1, 3}
    assert assignment_groups == {"1", "3"}
    assert len(result.selected_candidates) == 3


def test_spatial_group_topk_cli_writes_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    output_dir = tmp_path / "output"
    _candidate_rows().to_csv(candidates_csv, index=False)

    status = spatial_group_topk_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--output-dir",
            str(output_dir),
            "--group-top-k",
            "2",
            "--max-siblings-per-group",
            "1",
            "--diversity-weight",
            "1",
            "--diversity-scale-m",
            "1",
            "--score-column",
            "ranker_score",
            "--score-normalization",
            "none",
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
    selected = pd.read_csv(output_dir / "mmuad_spatial_group_topk_candidates.csv")
    summary = json.loads(
        (output_dir / "mmuad_spatial_group_topk_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected["mmuad_calibration_origin_row"].tolist() == [1, 3]
    assert summary["schema"] == "raft-uav-mmuad-spatial-group-topk-v1"
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_hypothesis_group_summary.json").exists()
