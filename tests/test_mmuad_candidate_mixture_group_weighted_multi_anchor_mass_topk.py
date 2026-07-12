from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk import (
    MultiAnchorAggregationConfig,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_multi_anchor_mass_topk import (
    AnchorReliabilityConfig,
    WEIGHTED_MULTI_ANCHOR_COST_COLUMN,
    WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN,
    _aggregate_weighted_anchor_costs,
    _resolve_anchor_weights,
    add_weighted_multi_anchor_conditioned_selection_utility,
    main as weighted_multi_anchor_main,
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


def test_weighted_mean_prefers_candidate_supported_by_reliable_anchor() -> None:
    scored, normalized, summary = add_weighted_multi_anchor_conditioned_selection_utility(
        _candidates(),
        {"reliable": _anchor(0.0), "weak": _anchor(10.0)},
        anchor_reliability={"reliable": 9.0, "weak": 1.0},
        mixture_config=_mixture_config(),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=1.0,
            anchor_scale_m=1.0,
            anchor_huber_delta=1.0,
            anchor_cost_cap=20.0,
            anchor_time_tolerance_s=0.01,
        ),
        aggregation_config=MultiAnchorAggregationConfig(aggregation="mean"),
    )

    left = scored.loc[scored["track_id"].astype(str).str.startswith("left-")]
    right = scored.loc[scored["track_id"].astype(str).str.startswith("right-")]
    assert left[WEIGHTED_MULTI_ANCHOR_COST_COLUMN].mean() < right[
        WEIGHTED_MULTI_ANCHOR_COST_COLUMN
    ].mean()
    assert left[WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN].mean() > right[
        WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN
    ].mean()
    assert set(normalized.groupby("anchor_name")["anchor_reliability_weight"].first()) == {
        1.0,
        9.0,
    }
    assert summary["anchor_reliability"] == {"reliable": 9.0, "weak": 1.0}
    assert summary["truth_used_for_weighting"] is False


def test_zero_weight_anchor_is_excluded_from_minimum() -> None:
    costs = np.asarray([[5.0, 0.0], [np.nan, 2.0]], dtype=float)

    aggregate, matched_weight, effective_count = _aggregate_weighted_anchor_costs(
        costs,
        anchor_weights=np.asarray([1.0, 0.0]),
        aggregation_config=MultiAnchorAggregationConfig(aggregation="minimum"),
    )

    assert aggregate.tolist() == [5.0, 0.0]
    assert matched_weight.tolist() == [1.0, 0.0]
    assert effective_count.tolist() == [1.0, 0.0]


def test_weighted_softmin_matches_weighted_logsumexp() -> None:
    costs = np.asarray([[0.0, 2.0]], dtype=float)
    weights = np.asarray([3.0, 1.0], dtype=float)
    temperature = 0.5

    aggregate, _, _ = _aggregate_weighted_anchor_costs(
        costs,
        anchor_weights=weights,
        aggregation_config=MultiAnchorAggregationConfig(
            aggregation="softmin",
            softmin_temperature=temperature,
        ),
    )

    expected = -temperature * np.log(
        0.75 * np.exp(-0.0 / temperature) + 0.25 * np.exp(-2.0 / temperature)
    )
    assert aggregate[0] == pytest.approx(expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_anchor_reliability_rejects_invalid_weights(value: float) -> None:
    with pytest.raises(ValueError):
        _resolve_anchor_weights(
            ["a", "b"],
            anchor_reliability={"a": value},
            default_weight=1.0,
        )


def test_weighted_multi_anchor_cli_writes_reliability_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _anchor(0.0).to_csv(left_csv, index=False)
    _anchor(10.0).to_csv(right_csv, index=False)

    status = weighted_multi_anchor_main(
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
            "--aggregation",
            "mean",
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
    summary_path = output_dir / "mmuad_weighted_multi_anchor_summary.json"
    scored_path = output_dir / "mmuad_weighted_multi_anchor_scored_candidates.csv"
    assert summary_path.exists()
    assert scored_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    reliability = payload["weighted_multi_anchor_conditioning"]["anchor_reliability"]
    assert reliability == {"left": 4.0, "right": 1.0}
    scored = pd.read_csv(scored_path)
    assert WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN in scored.columns
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()


def test_reliability_config_rejects_negative_default() -> None:
    with pytest.raises(ValueError):
        add_weighted_multi_anchor_conditioned_selection_utility(
            _candidates(),
            {"left": _anchor(0.0)},
            reliability_config=AnchorReliabilityConfig(default_weight=-1.0),
        )
