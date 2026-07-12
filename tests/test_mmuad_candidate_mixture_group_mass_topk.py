from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    main as posterior_mass_group_topk_main,
    run_posterior_mass_group_topk_candidate_mixture_map,
    select_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_spatial_topk import (
    SpatialHypothesisGroupTopKConfig,
    select_spatial_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def _candidate_rows() -> pd.DataFrame:
    records = []
    for time_s, scores in ((0.0, [10.0, 0.0, 0.0, 0.0]), (1.0, [0.0] * 4)):
        for index, score in enumerate(scores):
            records.append(
                {
                    "sequence_id": "seqA",
                    "time_s": time_s,
                    "source": f"source-{index}",
                    "track_id": f"frame-{time_s}-group-{index}",
                    "candidate_branch": "raw",
                    "mmuad_calibration_origin_row": int(time_s * 10) + index,
                    "x_m": float(index * 10),
                    "y_m": float(time_s),
                    "z_m": 0.0,
                    "ranker_score": score,
                    "predicted_sigma_m": 1.0,
                }
            )
    return pd.DataFrame.from_records(records)


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


def test_posterior_mass_budget_expands_on_ambiguous_frames() -> None:
    selected, summary = select_posterior_mass_hypothesis_group_topk(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=PosteriorMassGroupTopKConfig(
            min_group_top_k=1,
            max_group_top_k=4,
            target_posterior_mass=0.90,
            posterior_temperature=1.0,
            uniform_posterior_blend=0.0,
            max_siblings_per_group=1,
            diversity_weight=0.0,
        ),
    )

    budgets = selected.groupby("time_s")["mixture_mass_group_budget"].first().to_dict()
    assert budgets == {0.0: 1, 1.0: 4}
    assert len(selected.loc[selected["time_s"] == 0.0]) == 1
    assert len(selected.loc[selected["time_s"] == 1.0]) == 4
    assert summary["selected_group_budget_min"] == 1.0
    assert summary["selected_group_budget_max"] == 4.0


def test_fixed_mass_budget_matches_spatial_group_topk() -> None:
    rows = _candidate_rows().loc[lambda frame: frame["time_s"] == 0.0].copy()
    adaptive, _ = select_posterior_mass_hypothesis_group_topk(
        rows,
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=PosteriorMassGroupTopKConfig(
            min_group_top_k=2,
            max_group_top_k=2,
            target_posterior_mass=0.95,
            uniform_posterior_blend=0.0,
            max_siblings_per_group=1,
            diversity_weight=1.0,
            diversity_scale_m=1.0,
        ),
    )
    spatial, _ = select_spatial_hypothesis_group_topk(
        rows,
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=SpatialHypothesisGroupTopKConfig(
            group_top_k=2,
            max_siblings_per_group=1,
            diversity_weight=1.0,
            diversity_scale_m=1.0,
        ),
    )

    assert adaptive["track_id"].tolist() == spatial["track_id"].tolist()


def test_adaptive_group_topk_disables_second_row_truncation() -> None:
    result = run_posterior_mass_group_topk_candidate_mixture_map(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=PosteriorMassGroupTopKConfig(
            min_group_top_k=1,
            max_group_top_k=4,
            target_posterior_mass=0.90,
            uniform_posterior_blend=0.0,
            max_siblings_per_group=1,
            diversity_weight=0.0,
        ),
    )

    selected_at_ambiguous_frame = result.selected_candidates.loc[
        result.selected_candidates["time_s"] == 1.0
    ]
    assigned_at_ambiguous_frame = result.grouped_result.mixture_result.assignments.loc[
        result.grouped_result.mixture_result.assignments["time_s"] == 1.0
    ]
    assert len(selected_at_ambiguous_frame) == 4
    assert assigned_at_ambiguous_frame["mixture_hypothesis_group"].nunique() == 4


def test_disabled_adaptive_selection_returns_all_candidates() -> None:
    selected, summary = select_posterior_mass_hypothesis_group_topk(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        selection_config=PosteriorMassGroupTopKConfig(
            min_group_top_k=0,
            max_group_top_k=0,
        ),
    )

    assert len(selected) == len(_candidate_rows())
    assert not selected["mixture_mass_group_topk_selected"].any()
    assert summary["enabled"] is False


def test_posterior_mass_group_topk_cli_writes_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    output_dir = tmp_path / "output"
    _candidate_rows().to_csv(candidates_csv, index=False)

    status = posterior_mass_group_topk_main(
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
            "0.9",
            "--uniform-posterior-blend",
            "0",
            "--max-siblings-per-group",
            "1",
            "--diversity-weight",
            "0",
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
    selected = pd.read_csv(output_dir / "mmuad_posterior_mass_group_topk_candidates.csv")
    summary = json.loads(
        (output_dir / "mmuad_posterior_mass_group_topk_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected.groupby("time_s")["mixture_mass_group_budget"].first().to_dict() == {
        0.0: 1,
        1.0: 4,
    }
    assert summary["schema"] == "raft-uav-mmuad-posterior-mass-group-topk-v1"
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_hypothesis_group_summary.json").exists()
