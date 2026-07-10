from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_branch_uncertainty import (
    apply_branch_aware_candidate_uncertainty,
    attach_branch_class_uncertainty_context,
    attach_branch_uncertainty_context,
    train_branch_aware_candidate_uncertainty,
)
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    records = []
    for sequence_id, shift in (("seqA", 0.0), ("seqB", 1.0)):
        for time_s in (0.0, 1.0, 2.0):
            records.extend(
                [
                    {
                        "sequence_id": sequence_id,
                        "time_s": time_s,
                        "source": "lidar_360",
                        "track_id": f"{sequence_id}-raw-{time_s}",
                        "candidate_branch": "raw_static",
                        "x_m": time_s + shift,
                        "y_m": 0.0,
                        "z_m": 2.0,
                        "original_x_m": time_s + shift,
                        "original_y_m": 0.0,
                        "original_z_m": 2.0,
                        "ranker_score": 0.6,
                        "confidence": 0.6,
                        "cluster_point_count": 10,
                        "cluster_extent_3d_m": 1.0,
                    },
                    {
                        "sequence_id": sequence_id,
                        "time_s": time_s,
                        "source": "lidar_360",
                        "track_id": f"{sequence_id}-translated-{time_s}",
                        "candidate_branch": "source_translated_dynamic",
                        "x_m": time_s + shift + 4.0,
                        "y_m": 0.0,
                        "z_m": 2.0,
                        "original_x_m": time_s + shift + 9.0,
                        "original_y_m": 0.0,
                        "original_z_m": 2.0,
                        "ranker_score": 0.9,
                        "confidence": 0.9,
                        "cluster_point_count": 5,
                        "cluster_extent_3d_m": 2.0,
                    },
                    {
                        "sequence_id": sequence_id,
                        "time_s": time_s,
                        "source": "cross_sensor_merged",
                        "track_id": f"{sequence_id}-merged-{time_s}",
                        "candidate_branch": "cross_sensor_merged",
                        "x_m": time_s + shift + 1.0,
                        "y_m": 0.0,
                        "z_m": 2.0,
                        "ranker_score": 0.7,
                        "confidence": 0.7,
                        "cluster_point_count": 15,
                        "cluster_extent_3d_m": 0.5,
                    },
                ]
            )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    records = []
    for sequence_id, shift in (("seqA", 0.0), ("seqB", 1.0)):
        for time_s in (0.0, 1.0, 2.0):
            records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": time_s,
                    "x_m": time_s + shift,
                    "y_m": 0.0,
                    "z_m": 2.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _class_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "class_prob_0": [0.9, 0.1],
            "class_prob_1": [0.1, 0.9],
            "class_prob_2": [0.0, 0.0],
            "class_prob_3": [0.0, 0.0],
        }
    )


def test_branch_context_adds_semantics_translation_and_within_branch_ranks() -> None:
    contextual = attach_branch_uncertainty_context(CandidateFrame(_candidate_rows())).rows

    translated = contextual.loc[
        contextual["candidate_branch"] == "source_translated_dynamic"
    ]
    merged = contextual.loc[contextual["candidate_branch"] == "cross_sensor_merged"]
    raw = contextual.loc[contextual["candidate_branch"] == "raw_static"]

    assert (translated["candidate_reservoir_branch_is_translated"] == 1.0).all()
    assert (translated["candidate_reservoir_branch_is_dynamic"] == 1.0).all()
    assert np.allclose(translated["candidate_reservoir_translation_distance_m"], 5.0)
    assert (merged["candidate_reservoir_branch_is_merged"] == 1.0).all()
    assert (raw["candidate_reservoir_branch_is_raw"] == 1.0).all()
    assert (raw["candidate_reservoir_branch_is_static"] == 1.0).all()
    assert (contextual["candidate_reservoir_frame_branch_count"] == 3.0).all()
    assert (contextual["candidate_reservoir_branch_candidate_count"] == 1.0).all()
    assert (contextual["candidate_reservoir_branch_score_rank"] == 1.0).all()


def test_branch_class_context_adds_soft_interactions() -> None:
    contextual = attach_branch_class_uncertainty_context(
        CandidateFrame(_candidate_rows()),
        _class_probabilities(),
    ).rows

    column = "image_class_prob_0_x_candidate_reservoir_branch_is_translated"
    assert column in contextual.columns
    seq_a_translated = contextual.loc[
        (contextual["sequence_id"] == "seqA")
        & (contextual["candidate_branch"] == "source_translated_dynamic"),
        column,
    ]
    seq_b_translated = contextual.loc[
        (contextual["sequence_id"] == "seqB")
        & (contextual["candidate_branch"] == "source_translated_dynamic"),
        column,
    ]
    assert np.allclose(seq_a_translated, 0.9)
    assert np.allclose(seq_b_translated, 0.1)


def test_branch_aware_uncertainty_train_and_apply_without_test_truth() -> None:
    model, features, summary = train_branch_aware_candidate_uncertainty(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        _class_probabilities(),
        model_type="ridge",
        sigma_min_m=0.5,
        sigma_max_m=20.0,
        ridge_alpha=1.0,
        max_truth_time_delta_s=0.1,
    )

    assert any(
        column.startswith("candidate_reservoir_branch_is_")
        for column in model.feature_columns
    )
    assert any(
        "_x_candidate_reservoir_branch_is_translated" in column
        for column in model.feature_columns
    )
    assert summary["row_count"] == len(features)

    applied = apply_branch_aware_candidate_uncertainty(
        CandidateFrame(_candidate_rows()),
        model,
        _class_probabilities(),
        output_column="predicted_sigma_m_branch_class",
    ).rows
    sigma = pd.to_numeric(applied["predicted_sigma_m_branch_class"], errors="coerce")
    assert len(applied) == len(_candidate_rows())
    assert np.isfinite(sigma).all()
    assert (sigma >= 0.5).all()
    assert (sigma <= 20.0).all()
