from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_mass_topk import PosteriorMassGroupTopKConfig
from raft_uav.mmuad.candidate_mixture_group_spatial_mass_topk import (
    _spatial_order_mass_budget,
    main as spatial_mass_group_topk_main,
    run_spatial_posterior_mass_group_topk_candidate_mixture_map,
    select_spatial_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_spatial_topk import (
    SpatialHypothesisGroupTopKConfig,
    select_spatial_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def _mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        score_normalization="none",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
        loss="squared",
        smoothness_weight=0.0,
        iterations=1,
    )


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0] * 4,
            "source": ["lidar"] * 4,
            "track_id": ["A", "B", "C", "D"],
            "candidate_branch": ["raw", "raw", "dynamic", "translated"],
            "mmuad_calibration_origin_row": [0, 1, 2, 3],
            "x_m": [0.0, 0.1, 100.0, 200.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "ranker_score": [4.0, 3.0, 0.0, -1.0],
            "predicted_sigma_m": [1.0] * 4,
        }
    )


def test_spatial_order_budget_expands_until_actual_mass_reaches_target() -> None:
    groups = pd.DataFrame(
        {
            "mixture_hypothesis_group": ["A", "B", "C"],
            "mixture_spatial_group_score": [4.0, 3.0, 0.0],
        }
    )
    config = PosteriorMassGroupTopKConfig(
        min_group_top_k=1,
        max_group_top_k=3,
        target_posterior_mass=0.95,
        posterior_temperature=1.0,
        uniform_posterior_blend=0.0,
    )

    budget = _spatial_order_mass_budget(
        groups,
        ordered_group_ids=["A", "C", "B"],
        selection_config=config,
    )

    assert budget["ideal_selected_group_budget"] == 2
    assert budget["selected_group_budget"] == 3
    assert budget["budget_expansion_vs_score_order"] == 1
    assert budget["target_posterior_mass_reached"] is True
    assert budget["retained_posterior_mass"] >= 0.95


def test_fixed_budget_matches_existing_spatial_selector() -> None:
    rows = _candidate_rows()
    selection = PosteriorMassGroupTopKConfig(
        min_group_top_k=2,
        max_group_top_k=2,
        target_posterior_mass=0.95,
        uniform_posterior_blend=0.0,
        max_siblings_per_group=1,
        diversity_weight=2.0,
        diversity_scale_m=5.0,
    )

    corrected, _ = select_spatial_posterior_mass_hypothesis_group_topk(
        rows,
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=selection,
    )
    spatial, _ = select_spatial_hypothesis_group_topk(
        rows,
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=SpatialHypothesisGroupTopKConfig(
            group_top_k=2,
            max_siblings_per_group=1,
            diversity_weight=2.0,
            diversity_scale_m=5.0,
        ),
    )

    assert corrected["track_id"].tolist() == spatial["track_id"].tolist()
    assert corrected["mixture_spatial_mass_group_budget"].unique().tolist() == [2]


def test_corrected_selection_reports_actual_mass_and_disables_row_topk() -> None:
    result = run_spatial_posterior_mass_group_topk_candidate_mixture_map(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=PosteriorMassGroupTopKConfig(
            min_group_top_k=1,
            max_group_top_k=4,
            target_posterior_mass=0.99,
            uniform_posterior_blend=0.0,
            max_siblings_per_group=1,
            diversity_weight=2.0,
            diversity_scale_m=5.0,
        ),
    )

    selected = result.selected_candidates
    assigned = result.grouped_result.mixture_result.assignments
    assert selected["mixture_spatial_mass_group_retained_posterior_mass"].notna().all()
    assert selected["mixture_spatial_mass_group_target_reached"].all()
    assert assigned["mixture_hypothesis_group"].nunique() == selected[
        "mixture_hypothesis_group"
    ].nunique()
    assert result.selection_summary["truth_used_for_group_budget"] is False


def test_spatial_mass_group_topk_cli_writes_corrected_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    output_dir = tmp_path / "output"
    _candidate_rows().to_csv(candidates_csv, index=False)

    status = spatial_mass_group_topk_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--output-dir",
            str(output_dir),
            "--min-group-top-k",
            "1",
            "--max-group-top-k",
            "4",
            "--target-posterior-mass",
            "0.95",
            "--uniform-posterior-blend",
            "0",
            "--max-siblings-per-group",
            "1",
            "--diversity-weight",
            "2",
            "--diversity-scale-m",
            "5",
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
    selected = pd.read_csv(
        output_dir / "mmuad_spatial_posterior_mass_group_topk_candidates.csv"
    )
    summary = json.loads(
        (output_dir / "mmuad_spatial_posterior_mass_group_topk_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert "mixture_spatial_mass_group_retained_posterior_mass" in selected.columns
    assert summary["schema"] == "raft-uav-mmuad-spatial-posterior-mass-group-topk-v1"
    assert "target_posterior_mass_reached_fraction" in summary
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
