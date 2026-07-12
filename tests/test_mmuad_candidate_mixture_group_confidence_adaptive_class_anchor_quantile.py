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
    ClassConditionedAnchorReliabilityConfig,
    resolve_anchor_class_reliability,
)
from raft_uav.mmuad.candidate_mixture_group_confidence_adaptive_class_anchor_quantile import (
    ADAPTIVE_COST_COLUMN,
    ADAPTIVE_UTILITY_COLUMN,
    CONFIDENCE_SCORE_COLUMN,
    EFFECTIVE_STRENGTH_COLUMN,
    ConfidenceAdaptiveClassConditioningConfig,
    add_confidence_adaptive_class_anchor_quantile_selection_utility,
    class_probability_confidence,
    confidence_adaptive_class_conditioned_anchor_weight_matrix,
    main as confidence_adaptive_main,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig


def _candidates() -> pd.DataFrame:
    records = []
    for sequence_id in ("seq_confident", "seq_uniform"):
        for time_s in range(3):
            for branch, source, offset in (
                ("raw", "lidar_360", 0.0),
                ("translated", "livox_avia", 10.0),
            ):
                records.append(
                    {
                        "sequence_id": sequence_id,
                        "time_s": float(time_s),
                        "source": source,
                        "track_id": f"{sequence_id}-{branch}-{time_s}",
                        "origin_row": f"{sequence_id}-{branch}-origin-{time_s}",
                        "candidate_branch": branch,
                        "x_m": float(time_s + offset),
                        "y_m": 0.0,
                        "z_m": 1.0,
                        "ranker_score": 0.0,
                        "predicted_sigma_m": 1.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def _anchor(offset_m: float) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "Sequence": sequence_id,
                "Timestamp": float(time_s),
                "x_m": float(time_s + offset_m),
                "y_m": 0.0,
                "z_m": 1.0,
            }
            for sequence_id in ("seq_confident", "seq_uniform")
            for time_s in range(3)
        ]
    )


def _class_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq_confident", "seq_uniform"],
            "class_prob_0": [1.0, 0.25],
            "class_prob_1": [0.0, 0.25],
            "class_prob_2": [0.0, 0.25],
            "class_prob_3": [0.0, 0.25],
        }
    )


def _class_reliability() -> dict[str, dict[str, float]]:
    return {
        "left": {"0": 9.0, "1": 1.0, "2": 1.0, "3": 1.0},
        "right": {"0": 1.0, "1": 9.0, "2": 9.0, "3": 9.0},
    }


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


def test_entropy_confidence_has_expected_endpoints() -> None:
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    normalized, confidence, missing = class_probability_confidence(
        probabilities,
        mode="entropy",
    )

    assert normalized[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert confidence.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert missing.tolist() == [False, False, True]


def test_adaptive_weights_back_off_for_uniform_posterior() -> None:
    labels = ["left", "right"]
    base_weights = {"left": 1.0, "right": 1.0}
    class_weights = resolve_anchor_class_reliability(
        labels,
        base_weights=base_weights,
        anchor_class_reliability=_class_reliability(),
    )

    weights, fallback, confidence, effective_strength = (
        confidence_adaptive_class_conditioned_anchor_weight_matrix(
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                ],
                dtype=float,
            ),
            labels=labels,
            base_weights=base_weights,
            class_weights=class_weights,
            conditioning_strength=1.0,
            confidence_config=ConfidenceAdaptiveClassConditioningConfig(),
        )
    )

    assert weights[0].tolist() == [9.0, 1.0]
    assert weights[1].tolist() == [1.0, 1.0]
    assert fallback.tolist() == [False, False]
    assert confidence.tolist() == pytest.approx([1.0, 0.0])
    assert effective_strength.tolist() == pytest.approx([1.0, 0.0])


def test_confidence_floor_retains_bounded_conditioning() -> None:
    labels = ["left", "right"]
    base_weights = {"left": 1.0, "right": 1.0}
    class_weights = resolve_anchor_class_reliability(
        labels,
        base_weights=base_weights,
        anchor_class_reliability=_class_reliability(),
    )

    weights, _, _, effective_strength = (
        confidence_adaptive_class_conditioned_anchor_weight_matrix(
            np.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=float),
            labels=labels,
            base_weights=base_weights,
            class_weights=class_weights,
            conditioning_strength=0.8,
            confidence_config=ConfidenceAdaptiveClassConditioningConfig(
                confidence_floor=0.25,
            ),
        )
    )

    assert effective_strength.tolist() == pytest.approx([0.2])
    assert weights[0].tolist() == pytest.approx([1.4, 2.2])


def test_adaptive_selection_uses_class_only_when_confident() -> None:
    scored, _, summary = (
        add_confidence_adaptive_class_anchor_quantile_selection_utility(
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
            confidence_config=ConfidenceAdaptiveClassConditioningConfig(),
        )
    )

    confident = scored.loc[scored["sequence_id"] == "seq_confident"]
    uniform = scored.loc[scored["sequence_id"] == "seq_uniform"]
    confident_left = confident.loc[
        confident["candidate_branch"] == "raw"
    ]
    confident_right = confident.loc[
        confident["candidate_branch"] == "translated"
    ]

    assert confident_left[ADAPTIVE_COST_COLUMN].mean() < confident_right[
        ADAPTIVE_COST_COLUMN
    ].mean()
    assert confident_left[ADAPTIVE_UTILITY_COLUMN].mean() > confident_right[
        ADAPTIVE_UTILITY_COLUMN
    ].mean()
    assert np.allclose(confident[CONFIDENCE_SCORE_COLUMN], 1.0)
    assert np.allclose(confident[EFFECTIVE_STRENGTH_COLUMN], 1.0)
    assert np.allclose(uniform[CONFIDENCE_SCORE_COLUMN], 0.0)
    assert np.allclose(uniform[EFFECTIVE_STRENGTH_COLUMN], 0.0)
    assert np.allclose(
        uniform["mixture_class_conditioned_anchor_weight_left"],
        1.0,
    )
    assert np.allclose(
        uniform["mixture_class_conditioned_anchor_weight_right"],
        1.0,
    )
    assert summary["truth_used_for_weighting"] is False
    assert summary["mean_effective_class_conditioning_strength"] == pytest.approx(0.5)


def test_confidence_adaptive_cli_writes_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    probabilities_csv = tmp_path / "class_probabilities.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _anchor(0.0).to_csv(left_csv, index=False)
    _anchor(10.0).to_csv(right_csv, index=False)
    _class_probabilities().to_csv(probabilities_csv, index=False)

    status = confidence_adaptive_main(
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
            "right:0=1",
            "--class-confidence-mode",
            "entropy",
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
    summary_path = output_dir / "mmuad_confidence_adaptive_class_summary.json"
    scored_path = output_dir / "mmuad_confidence_adaptive_class_scored_candidates.csv"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    conditioning = payload["confidence_adaptive_class_anchor_quantile"]
    assert conditioning["confidence_config"]["confidence_mode"] == "entropy"
    scored = pd.read_csv(scored_path)
    assert {CONFIDENCE_SCORE_COLUMN, EFFECTIVE_STRENGTH_COLUMN}.issubset(
        scored.columns
    )
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
