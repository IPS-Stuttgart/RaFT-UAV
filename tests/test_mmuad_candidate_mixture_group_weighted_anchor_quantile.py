from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_anchor_quantile import (
    WEIGHTED_QUANTILE_COST_COLUMN,
    WEIGHTED_QUANTILE_UTILITY_COLUMN,
    WeightedAnchorQuantileConfig,
    add_weighted_quantile_multi_anchor_conditioned_selection_utility,
    aggregate_weighted_quantile_anchor_costs,
    main as weighted_quantile_main,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig


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
        score_normalization="none",
        score_weight=0.0,
        temperature=1.0,
        sigma_log_weight=0.0,
        smoothness_weight=0.0,
        iterations=1,
    )


def test_weighted_quantile_interpolates_between_minimum_and_maximum() -> None:
    costs = np.asarray([[1.0, 9.0, 10.0]], dtype=float)
    weights = np.asarray([8.0, 1.0, 1.0], dtype=float)

    minimum, _, _ = aggregate_weighted_quantile_anchor_costs(
        costs,
        anchor_weights=weights,
        quantile=0.0,
    )
    median, matched_weight, effective_count = aggregate_weighted_quantile_anchor_costs(
        costs,
        anchor_weights=weights,
        quantile=0.5,
    )
    ninety_percent, _, _ = aggregate_weighted_quantile_anchor_costs(
        costs,
        anchor_weights=weights,
        quantile=0.9,
    )
    maximum, _, _ = aggregate_weighted_quantile_anchor_costs(
        costs,
        anchor_weights=weights,
        quantile=1.0,
    )

    assert minimum.tolist() == [1.0]
    assert median.tolist() == [1.0]
    assert ninety_percent.tolist() == [9.0]
    assert maximum.tolist() == [10.0]
    assert matched_weight.tolist() == [10.0]
    assert 1.0 < effective_count[0] < 3.0


def test_weighted_median_rejects_mode_supported_only_by_weak_anchor() -> None:
    scored, normalized, summary = (
        add_weighted_quantile_multi_anchor_conditioned_selection_utility(
            _candidates(),
            {"reliable": _anchor(0.0), "weak": _anchor(10.0)},
            anchor_reliability={"reliable": 4.0, "weak": 1.0},
            mixture_config=_mixture_config(),
            anchor_config=AnchorConditioningConfig(
                anchor_selection_weight=1.0,
                anchor_scale_m=1.0,
                anchor_huber_delta=1.0,
                anchor_cost_cap=20.0,
                anchor_time_tolerance_s=0.01,
            ),
            quantile_config=WeightedAnchorQuantileConfig(cost_quantile=0.5),
        )
    )

    left = scored.loc[scored["track_id"].astype(str).str.startswith("left-")]
    right = scored.loc[scored["track_id"].astype(str).str.startswith("right-")]
    assert left[WEIGHTED_QUANTILE_COST_COLUMN].mean() < right[
        WEIGHTED_QUANTILE_COST_COLUMN
    ].mean()
    assert left[WEIGHTED_QUANTILE_UTILITY_COLUMN].mean() > right[
        WEIGHTED_QUANTILE_UTILITY_COLUMN
    ].mean()
    assert set(normalized.groupby("anchor_name")["anchor_reliability_weight"].first()) == {
        1.0,
        4.0,
    }
    assert summary["weighted_aggregation"] == "quantile"
    assert summary["quantile_config"]["cost_quantile"] == 0.5
    assert summary["truth_used_for_weighting"] is False


@pytest.mark.parametrize("quantile", [float("nan"), float("inf"), -0.1, 1.1])
def test_weighted_quantile_rejects_invalid_values(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        aggregate_weighted_quantile_anchor_costs(
            np.asarray([[0.0, 1.0]], dtype=float),
            anchor_weights=np.asarray([1.0, 1.0], dtype=float),
            quantile=quantile,
        )


def test_weighted_quantile_cli_writes_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _anchor(0.0).to_csv(left_csv, index=False)
    _anchor(10.0).to_csv(right_csv, index=False)

    status = weighted_quantile_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--anchor-csv",
            f"left={left_csv}",
            "--anchor-csv",
            f"right={right_csv}",
            "--anchor-reliability",
            "left=4",
            "--anchor-reliability",
            "right=1",
            "--anchor-cost-quantile",
            "0.5",
            "--output-dir",
            str(output_dir),
            "--min-group-top-k",
            "1",
            "--max-group-top-k",
            "1",
            "--max-siblings-per-group",
            "1",
            "--diversity-weight",
            "0",
            "--anchor-selection-weight",
            "1",
            "--anchor-scale-m",
            "1",
            "--anchor-cost-cap",
            "20",
            "--anchor-time-tolerance-s",
            "0.01",
            "--score-column",
            "ranker_score",
            "--score-weight",
            "0",
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
    summary_path = output_dir / "mmuad_weighted_anchor_quantile_summary.json"
    scored_path = output_dir / "mmuad_weighted_anchor_quantile_scored_candidates.csv"
    assert summary_path.exists()
    assert scored_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    conditioning = payload["weighted_anchor_quantile_conditioning"]
    assert conditioning["anchor_reliability"] == {"left": 4.0, "right": 1.0}
    assert conditioning["quantile_config"]["cost_quantile"] == 0.5
    scored = pd.read_csv(scored_path)
    assert WEIGHTED_QUANTILE_UTILITY_COLUMN in scored.columns
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
