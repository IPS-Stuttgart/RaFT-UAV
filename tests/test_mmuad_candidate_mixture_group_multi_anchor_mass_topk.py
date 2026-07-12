from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk import (
    MultiAnchorConditioningConfig,
    add_multi_anchor_conditioned_selection_utility,
    main as multi_anchor_main,
    select_multi_anchor_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def _candidates() -> pd.DataFrame:
    records = []
    for time_s in range(3):
        for label, offset in (("left", 0.0), ("right", 10.0), ("middle", 5.0)):
            records.append(
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"{label}-{time_s}",
                    "origin_row": f"{label}-origin-{time_s}",
                    "candidate_branch": label,
                    "x_m": float(time_s) + offset,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 1.0,
                    "predicted_sigma_m": 1.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _anchor(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [offset, offset + 1.0, offset + 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0],
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


def _selection_config(top_k: int) -> PosteriorMassGroupTopKConfig:
    return PosteriorMassGroupTopKConfig(
        min_group_top_k=top_k,
        max_group_top_k=top_k,
        target_posterior_mass=0.95,
        posterior_temperature=1.0,
        uniform_posterior_blend=0.0,
        max_siblings_per_group=1,
        group_score_mode="max",
        diversity_weight=0.0,
        diversity_scale_m=5.0,
        diversity_cap_m=30.0,
    )


def _anchor_config() -> AnchorConditioningConfig:
    return AnchorConditioningConfig(
        anchor_selection_weight=5.0,
        anchor_scale_m=5.0,
        anchor_huber_delta=1.0,
        anchor_cost_cap=4.0,
        anchor_time_tolerance_s=0.01,
    )


def test_min_aggregation_preserves_candidates_supported_by_either_anchor() -> None:
    selected, final_anchor, scored, summary = (
        select_multi_anchor_posterior_mass_hypothesis_group_topk(
            _candidates(),
            initial_estimates_by_name={"left": _anchor(0.0), "right": _anchor(10.0)},
            mixture_config=_mixture_config(),
            group_config=HypothesisGroupConfig(group_column="origin_row"),
            selection_config=_selection_config(2),
            anchor_config=_anchor_config(),
            multi_anchor_config=MultiAnchorConditioningConfig(
                aggregation="min",
                final_anchor_policy="none",
            ),
        )
    )

    assert final_anchor is None
    assert set(selected["candidate_branch"]) == {"left", "right"}
    endpoint_costs = scored.loc[
        scored["candidate_branch"].isin(["left", "right"]),
        "mixture_multi_anchor_cost",
    ]
    middle_costs = scored.loc[
        scored["candidate_branch"] == "middle", "mixture_multi_anchor_cost"
    ]
    assert endpoint_costs.eq(0.0).all()
    assert middle_costs.gt(0.0).all()
    assert summary["multi_anchor_conditioning"]["anchor_count"] == 2
    assert summary["truth_used_for_selection"] is False


def test_mean_aggregation_rewards_anchor_consensus() -> None:
    selected, _, scored, _ = select_multi_anchor_posterior_mass_hypothesis_group_topk(
        _candidates(),
        initial_estimates_by_name={"left": _anchor(0.0), "right": _anchor(10.0)},
        mixture_config=_mixture_config(),
        group_config=HypothesisGroupConfig(group_column="origin_row"),
        selection_config=_selection_config(1),
        anchor_config=_anchor_config(),
        multi_anchor_config=MultiAnchorConditioningConfig(aggregation="mean"),
    )

    assert set(selected["candidate_branch"]) == {"middle"}
    middle = scored.loc[
        scored["candidate_branch"] == "middle", "mixture_multi_anchor_cost"
    ].mean()
    endpoint = scored.loc[
        scored["candidate_branch"] == "left", "mixture_multi_anchor_cost"
    ].mean()
    assert middle < endpoint


def test_median_final_anchor_is_built_at_candidate_times() -> None:
    _, _, final_anchor, summary = add_multi_anchor_conditioned_selection_utility(
        _candidates(),
        {"left": _anchor(0.0), "right": _anchor(10.0)},
        mixture_config=_mixture_config(),
        anchor_config=_anchor_config(),
        multi_anchor_config=MultiAnchorConditioningConfig(
            aggregation="softmin",
            final_anchor_policy="median",
        ),
    )

    assert final_anchor is not None
    assert final_anchor["state_x_m"].tolist() == [5.0, 6.0, 7.0]
    assert summary["final_initial_estimate_rows"] == 3


def test_multi_anchor_config_rejects_invalid_softmin_temperature() -> None:
    with pytest.raises(ValueError, match="softmin_temperature"):
        add_multi_anchor_conditioned_selection_utility(
            _candidates(),
            {"left": _anchor(0.0)},
            mixture_config=_mixture_config(),
            multi_anchor_config=MultiAnchorConditioningConfig(
                aggregation="softmin",
                softmin_temperature=0.0,
            ),
        )


def test_multi_anchor_cli_writes_diagnostics_and_grouped_outputs(tmp_path: Path) -> None:
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
            "--initial-estimates",
            f"left={left_csv}",
            "--initial-estimates",
            f"right={right_csv}",
            "--output-dir",
            str(output_dir),
            "--anchor-aggregation",
            "min",
            "--final-anchor-policy",
            "median",
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
            "5",
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
    scored_path = output_dir / "mmuad_multi_anchor_scored_candidates.csv"
    selected_path = output_dir / "mmuad_multi_anchor_selected_candidates.csv"
    summary_path = output_dir / "mmuad_multi_anchor_posterior_mass_group_topk_summary.json"
    final_anchor_path = output_dir / "mmuad_multi_anchor_final_initial_estimates.csv"
    assert scored_path.exists()
    assert selected_path.exists()
    assert summary_path.exists()
    assert final_anchor_path.exists()
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["multi_anchor_conditioning"]["anchor_names"] == ["left", "right"]
    assert summary["multi_anchor_conditioning"]["final_initial_estimate_rows"] == 3
