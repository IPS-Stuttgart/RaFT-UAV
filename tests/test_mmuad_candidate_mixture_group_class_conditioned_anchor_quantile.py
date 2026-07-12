from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_class_conditioned_anchor_quantile import (
    CLASS_CONDITIONED_COST_COLUMN,
    CLASS_CONDITIONED_UTILITY_COLUMN,
    ClassConditionedAnchorReliabilityConfig,
    _parse_anchor_class_reliability_specs,
    add_class_conditioned_anchor_quantile_selection_utility,
    aggregate_rowwise_weighted_quantile_anchor_costs,
    class_conditioned_anchor_weight_matrix,
    main as class_conditioned_main,
    resolve_anchor_class_reliability,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig


def _candidates() -> pd.DataFrame:
    records = []
    for sequence_id in ("seq0", "seq1"):
        for time_s in range(3):
            records.extend(
                [
                    {
                        "sequence_id": sequence_id,
                        "time_s": float(time_s),
                        "source": "lidar_360",
                        "track_id": f"{sequence_id}-left-{time_s}",
                        "origin_row": f"{sequence_id}-left-origin-{time_s}",
                        "candidate_branch": "raw",
                        "x_m": float(time_s),
                        "y_m": 0.0,
                        "z_m": 1.0,
                        "ranker_score": 0.0,
                        "predicted_sigma_m": 1.0,
                    },
                    {
                        "sequence_id": sequence_id,
                        "time_s": float(time_s),
                        "source": "livox_avia",
                        "track_id": f"{sequence_id}-right-{time_s}",
                        "origin_row": f"{sequence_id}-right-origin-{time_s}",
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
    records = []
    for sequence_id in ("seq0", "seq1"):
        for time_s in range(3):
            records.append(
                {
                    "Sequence": sequence_id,
                    "Timestamp": float(time_s),
                    "x_m": float(time_s + offset_m),
                    "y_m": 0.0,
                    "z_m": 1.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _class_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0", "seq1"],
            "class_prob_0": [1.0, 0.0],
            "class_prob_1": [0.0, 1.0],
            "class_prob_2": [0.0, 0.0],
            "class_prob_3": [0.0, 0.0],
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


def _class_reliability() -> dict[str, dict[str, float]]:
    return {
        "left": {"0": 9.0, "1": 1.0},
        "right": {"0": 1.0, "1": 9.0},
    }


def test_rowwise_weighted_quantile_uses_candidate_specific_weights() -> None:
    costs = np.asarray([[0.0, 10.0], [0.0, 10.0]], dtype=float)
    weights = np.asarray([[9.0, 1.0], [1.0, 9.0]], dtype=float)

    aggregate, matched, effective = aggregate_rowwise_weighted_quantile_anchor_costs(
        costs,
        anchor_weights=weights,
        quantile=0.5,
    )

    assert aggregate.tolist() == [0.0, 10.0]
    assert matched.tolist() == [10.0, 10.0]
    assert np.all((effective > 1.0) & (effective < 2.0))


def test_class_conditioned_weights_blend_soft_probabilities_and_fallback() -> None:
    labels = ["left", "right"]
    base = {"left": 1.0, "right": 1.0}
    class_weights = resolve_anchor_class_reliability(
        labels,
        base_weights=base,
        anchor_class_reliability=_class_reliability(),
    )
    probabilities = np.asarray(
        [
            [0.75, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    weights, fallback = class_conditioned_anchor_weight_matrix(
        probabilities,
        labels=labels,
        base_weights=base,
        class_weights=class_weights,
        conditioning_strength=0.5,
    )

    assert weights[0].tolist() == pytest.approx([4.0, 2.0])
    assert weights[1].tolist() == [1.0, 1.0]
    assert fallback.tolist() == [False, True]


def test_class_conditioned_quantile_prefers_different_modes_by_class() -> None:
    scored, normalized, summary = (
        add_class_conditioned_anchor_quantile_selection_utility(
            _candidates(),
            {"left": _anchor(0.0), "right": _anchor(10.0)},
            _class_probabilities(),
            anchor_reliability={"left": 1.0, "right": 1.0},
            anchor_class_reliability=_class_reliability(),
            mixture_config=_mixture_config(),
            anchor_config=AnchorConditioningConfig(
                anchor_selection_weight=1.0,
                anchor_scale_m=1.0,
                anchor_huber_delta=1.0,
                anchor_cost_cap=20.0,
                anchor_time_tolerance_s=0.01,
            ),
            reliability_config=ClassConditionedAnchorReliabilityConfig(
                cost_quantile=0.5,
                conditioning_strength=1.0,
            ),
        )
    )

    seq0_left = scored.loc[
        scored["track_id"].astype(str).str.startswith("seq0-left-")
    ]
    seq0_right = scored.loc[
        scored["track_id"].astype(str).str.startswith("seq0-right-")
    ]
    seq1_left = scored.loc[
        scored["track_id"].astype(str).str.startswith("seq1-left-")
    ]
    seq1_right = scored.loc[
        scored["track_id"].astype(str).str.startswith("seq1-right-")
    ]

    assert seq0_left[CLASS_CONDITIONED_COST_COLUMN].mean() < seq0_right[
        CLASS_CONDITIONED_COST_COLUMN
    ].mean()
    assert seq1_right[CLASS_CONDITIONED_COST_COLUMN].mean() < seq1_left[
        CLASS_CONDITIONED_COST_COLUMN
    ].mean()
    assert seq0_left[CLASS_CONDITIONED_UTILITY_COLUMN].mean() > seq0_right[
        CLASS_CONDITIONED_UTILITY_COLUMN
    ].mean()
    assert seq1_right[CLASS_CONDITIONED_UTILITY_COLUMN].mean() > seq1_left[
        CLASS_CONDITIONED_UTILITY_COLUMN
    ].mean()
    assert seq0_left["mixture_class_conditioned_anchor_weight_left"].iloc[0] == 9.0
    assert seq1_left["mixture_class_conditioned_anchor_weight_left"].iloc[0] == 1.0
    anchor_rows = normalized.groupby("anchor_name").first()
    assert anchor_rows.loc["left", "anchor_class_reliability_0"] == 9.0
    assert anchor_rows.loc["right", "anchor_class_reliability_1"] == 9.0
    assert summary["truth_used_for_weighting"] is False
    assert summary["class_conditioning_fallback_rows"] == 0


def test_parse_anchor_class_reliability_specs_validates_names_and_classes() -> None:
    parsed = _parse_anchor_class_reliability_specs(
        ["left:0=4", "left:1=2", "right:0=1"],
        {"left", "right"},
    )
    assert parsed == {
        "left": {"0": 4.0, "1": 2.0},
        "right": {"0": 1.0},
    }
    with pytest.raises(ValueError, match="unknown anchor"):
        _parse_anchor_class_reliability_specs(["missing:0=1"], {"left"})
    with pytest.raises(ValueError, match="unsupported Track 5 class"):
        _parse_anchor_class_reliability_specs(["left:9=1"], {"left"})


def test_class_conditioned_anchor_quantile_cli_writes_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    probabilities_csv = tmp_path / "class_probabilities.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _anchor(0.0).to_csv(left_csv, index=False)
    _anchor(10.0).to_csv(right_csv, index=False)
    _class_probabilities().to_csv(probabilities_csv, index=False)

    status = class_conditioned_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--anchor-csv",
            f"left={left_csv}",
            "--anchor-csv",
            f"right={right_csv}",
            "--class-probabilities-csv",
            str(probabilities_csv),
            "--anchor-class-reliability",
            "left:0=9",
            "--anchor-class-reliability",
            "left:1=1",
            "--anchor-class-reliability",
            "right:0=1",
            "--anchor-class-reliability",
            "right:1=9",
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
    summary_path = (
        output_dir / "mmuad_class_conditioned_anchor_quantile_summary.json"
    )
    scored_path = (
        output_dir
        / "mmuad_class_conditioned_anchor_quantile_scored_candidates.csv"
    )
    assert summary_path.exists()
    assert scored_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    conditioning = payload["class_conditioned_anchor_quantile"]
    assert conditioning["anchor_class_reliability"]["left"]["0"] == 9.0
    scored = pd.read_csv(scored_path)
    assert CLASS_CONDITIONED_UTILITY_COLUMN in scored.columns
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
