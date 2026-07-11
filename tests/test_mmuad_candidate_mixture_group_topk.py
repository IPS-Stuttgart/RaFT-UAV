from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_topk import (
    HypothesisGroupTopKConfig,
    main as group_topk_main,
    run_group_topk_candidate_mixture_map,
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
                "x_m": 0.2,
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
                "x_m": 5.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.80,
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


def test_group_topk_prevents_siblings_from_consuming_all_slots() -> None:
    selected, summary = select_hypothesis_group_topk(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=HypothesisGroupTopKConfig(
            group_top_k=2,
            max_siblings_per_group=1,
        ),
    )

    assert selected["mmuad_calibration_origin_row"].tolist() == [1, 2]
    assert selected["mixture_group_topk_group_rank"].tolist() == [1, 2]
    assert summary["selected_candidate_rows"] == 2
    assert summary["selected_groups_per_frame_mean"] == 2.0


def test_group_topk_can_keep_raw_and_calibrated_siblings() -> None:
    selected, _ = select_hypothesis_group_topk(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        selection_config=HypothesisGroupTopKConfig(
            group_top_k=1,
            max_siblings_per_group=2,
        ),
    )

    assert selected["mmuad_calibration_origin_row"].tolist() == [1, 1]
    assert set(selected["candidate_branch"]) == {
        "raw",
        "source_translation_calibrated",
    }
    assert selected["mixture_group_topk_sibling_rank"].tolist() == [1, 2]


def test_group_topk_run_disables_second_row_level_truncation() -> None:
    result = run_group_topk_candidate_mixture_map(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=HypothesisGroupTopKConfig(
            group_top_k=2,
            max_siblings_per_group=2,
        ),
    )

    selected_groups = set(result.selected_candidates["mmuad_calibration_origin_row"])
    assignment_groups = set(
        result.grouped_result.mixture_result.assignments["mixture_hypothesis_group"]
    )
    assert selected_groups == {1, 2}
    assert assignment_groups == {"1", "2"}
    assert len(result.selected_candidates) == 3


def test_group_topk_cli_writes_selection_and_mixture_artifacts(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    output_dir = tmp_path / "output"
    _candidate_rows().to_csv(candidates_csv, index=False)

    status = group_topk_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--output-dir",
            str(output_dir),
            "--group-top-k",
            "2",
            "--max-siblings-per-group",
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
    selected = pd.read_csv(output_dir / "mmuad_group_topk_candidates.csv")
    summary = json.loads(
        (output_dir / "mmuad_group_topk_summary.json").read_text(encoding="utf-8")
    )
    assert selected["mmuad_calibration_origin_row"].tolist() == [1, 2]
    assert summary["selected_candidate_rows"] == 2
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_hypothesis_group_summary.json").exists()
