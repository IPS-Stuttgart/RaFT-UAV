from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk import (
    MultiAnchorAggregationConfig,
    add_multi_anchor_conditioned_selection_utility,
    main as multi_anchor_main,
    select_multi_anchor_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def _candidates() -> pd.DataFrame:
    records = []
    for time_s in range(3):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"left-{time_s}",
                    "origin_row": f"left-origin-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.0,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"middle-{time_s}",
                    "origin_row": f"middle-origin-{time_s}",
                    "candidate_branch": "dynamic",
                    "x_m": float(time_s + 5),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 1.0,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "radar_enhance_pcl",
                    "track_id": f"right-{time_s}",
                    "origin_row": f"right-origin-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.0,
                    "predicted_sigma_m": 1.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _anchor(offset_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqA"],
            "Timestamp": [0.0, 1.0, 2.0],
            "x_m": [offset_m, 1.0 + offset_m, 2.0 + offset_m],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def _mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        fallback_score_columns=(),
        sigma_column="predicted_sigma_m",
        score_normalization="minmax",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
        smoothness_weight=0.0,
        iterations=1,
    )


def _selection_config(group_top_k: int = 2) -> PosteriorMassGroupTopKConfig:
    return PosteriorMassGroupTopKConfig(
        min_group_top_k=group_top_k,
        max_group_top_k=group_top_k,
        target_posterior_mass=0.95,
        posterior_temperature=1.0,
        uniform_posterior_blend=0.0,
        max_siblings_per_group=1,
        group_score_mode="max",
        diversity_weight=0.0,
        diversity_scale_m=5.0,
        diversity_cap_m=30.0,
    )


def test_minimum_aggregation_preserves_candidates_coherent_with_either_anchor() -> None:
    scored, normalized, summary = add_multi_anchor_conditioned_selection_utility(
        _candidates(),
        {"left_path": _anchor(0.0), "right_path": _anchor(10.0)},
        mixture_config=_mixture_config(),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=5.0,
            anchor_scale_m=1.0,
            anchor_huber_delta=1.0,
            anchor_cost_cap=10.0,
            anchor_time_tolerance_s=0.01,
        ),
        aggregation_config=MultiAnchorAggregationConfig(aggregation="minimum"),
    )

    left = scored.loc[scored["track_id"].astype(str).str.startswith("left-")]
    right = scored.loc[scored["track_id"].astype(str).str.startswith("right-")]
    middle = scored.loc[scored["track_id"].astype(str).str.startswith("middle-")]
    assert (left["mixture_multi_anchor_aggregate_cost"] == 0.0).all()
    assert (right["mixture_multi_anchor_aggregate_cost"] == 0.0).all()
    assert (middle["mixture_multi_anchor_aggregate_cost"] > 0.0).all()
    assert set(left["mixture_multi_anchor_best_anchor"]) == {"left_path"}
    assert set(right["mixture_multi_anchor_best_anchor"]) == {"right_path"}
    assert normalized["anchor_name"].nunique() == 2
    assert summary["anchor_count"] == 2
    assert summary["matched_frame_fraction"] == 1.0


def test_multi_anchor_group_selector_retains_both_trajectory_modes() -> None:
    _, selected, _, summary = select_multi_anchor_posterior_mass_hypothesis_group_topk(
        _candidates(),
        anchor_estimates={"left": _anchor(0.0), "right": _anchor(10.0)},
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(group_column="origin_row"),
        selection_config=_selection_config(group_top_k=2),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=5.0,
            anchor_scale_m=1.0,
            anchor_cost_cap=10.0,
            anchor_time_tolerance_s=0.01,
        ),
    )

    selected_prefix = selected["track_id"].astype(str).str.split("-").str[0]
    assert set(selected_prefix) == {"left", "right"}
    assert summary["truth_used_for_selection"] is False
    assert summary["multi_anchor_conditioning"]["anchor_count"] == 2


def test_multi_anchor_cli_writes_full_scoring_and_grouped_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _anchor(0.0).to_csv(left_csv, index=False)
    _anchor(10.0).to_csv(right_csv, index=False)

    status = multi_anchor_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--anchor-csv",
            f"left={left_csv}",
            "--anchor-csv",
            f"right={right_csv}",
            "--output-dir",
            str(output_dir),
            "--min-group-top-k",
            "2",
            "--max-group-top-k",
            "2",
            "--max-siblings-per-group",
            "1",
            "--diversity-weight",
            "0",
            "--anchor-selection-weight",
            "5",
            "--anchor-scale-m",
            "1",
            "--anchor-cost-cap",
            "10",
            "--anchor-time-tolerance-s",
            "0.01",
            "--score-column",
            "ranker_score",
            "--sigma-log-weight",
            "0",
            "--smoothness-weight",
            "0",
            "--iterations",
            "1",
            "--hypothesis-group-column",
            "origin_row",
        ]
    )

    assert status == 0
    scored_path = output_dir / "mmuad_multi_anchor_posterior_mass_scored_candidates.csv"
    selected_path = output_dir / "mmuad_multi_anchor_posterior_mass_selected_candidates.csv"
    anchors_path = output_dir / "mmuad_multi_anchor_normalized_anchors.csv"
    summary_path = output_dir / "mmuad_multi_anchor_posterior_mass_summary.json"
    assert scored_path.exists()
    assert selected_path.exists()
    assert anchors_path.exists()
    assert summary_path.exists()
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["multi_anchor_conditioning"]["anchor_labels"] == ["left", "right"]
    assert payload["final_initial_estimates_supplied"] is False
