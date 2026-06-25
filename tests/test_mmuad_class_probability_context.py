from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context
from raft_uav.mmuad.class_probability_context import main as context_main
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table, train_cluster_ranker
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["lidar_360", "livox_avia", "lidar_360"],
            "track_id": ["a", "b", "c"],
            "candidate_branch": ["static_dynamic_union", "source_translation", "static_dynamic_union"],
            "x_m": [1.0, 2.0, 3.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
            "confidence": [0.5, 0.7, 0.9],
            "cluster_point_count": [10, 20, 30],
            "cluster_extent_3d_m": [1.0, 2.0, 3.0],
        }
    )


def _probability_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "predicted_probability_0": [0.1, 0.7],
            "predicted_probability_1": [0.2, 0.1],
            "predicted_probability_2": [0.6, 0.1],
            "predicted_probability_3": [0.1, 0.1],
            "predicted_class": [2, 0],
        }
    )


def test_class_probability_context_adds_ranker_consumable_features() -> None:
    augmented = attach_class_probability_context(
        CandidateFrame(_candidate_rows()),
        _probability_rows(),
        interaction_columns=(
            "cluster_point_count",
            "cluster_extent_3d_m",
            "image_candidate_branch_dynamic_flag",
            "image_source_is_lidar_360",
        ),
    )
    rows = augmented.rows.sort_values(["sequence_id", "track_id"]).reset_index(drop=True)

    assert "image_class_prob_2" in rows.columns
    assert "image_class_entropy" in rows.columns
    assert "image_candidate_branch_dynamic_flag" in rows.columns
    assert "image_source_is_lidar_360" in rows.columns
    assert "image_class_prob_2_x_cluster_point_count" in rows.columns
    assert "image_class_prob_2_x_image_candidate_branch_dynamic_flag" in rows.columns
    assert "image_class_prob_2_x_image_source_is_lidar_360" in rows.columns
    assert rows.loc[0, "image_class_prob_2"] == pytest.approx(0.6)
    assert rows.loc[0, "image_candidate_branch_dynamic_flag"] == pytest.approx(1.0)
    assert rows.loc[1, "image_candidate_branch_dynamic_flag"] == pytest.approx(0.0)
    assert rows.loc[0, "image_class_prob_2_x_cluster_point_count"] == pytest.approx(6.0)
    assert rows.loc[0, "image_class_prob_2_x_image_candidate_branch_dynamic_flag"] == pytest.approx(0.6)
    assert rows.loc[0, "image_class_prob_2_x_image_source_is_lidar_360"] == pytest.approx(0.6)
    assert rows.loc[2, "image_predicted_class_0"] == pytest.approx(1.0)

    features = build_cluster_feature_table(augmented)
    model = train_cluster_ranker(
        features.assign(good_cluster=[True, False, True]),
        iterations=2,
        learning_rate=0.1,
    )
    assert "image_class_prob_2" in model.feature_columns
    assert "image_class_prob_2_x_cluster_point_count" in model.feature_columns
    assert "image_class_prob_2_x_image_candidate_branch_dynamic_flag" in model.feature_columns


def test_class_probability_context_fills_missing_sequences_uniformly() -> None:
    probabilities = _probability_rows().loc[lambda frame: frame["sequence_id"] == "seqA"]
    augmented = attach_class_probability_context(
        CandidateFrame(_candidate_rows()),
        probabilities,
        interaction_columns=("cluster_point_count",),
    )
    seq_b = augmented.rows.loc[augmented.rows["sequence_id"] == "seqB"].iloc[0]

    assert seq_b["image_class_probability_available"] == pytest.approx(0.0)
    assert seq_b["image_class_prob_0"] == pytest.approx(0.25)
    assert seq_b["image_class_prob_3"] == pytest.approx(0.25)


def test_class_probability_context_can_reject_missing_sequences() -> None:
    probabilities = _probability_rows().loc[lambda frame: frame["sequence_id"] == "seqA"]
    with pytest.raises(ValueError, match="missing class probabilities"):
        attach_class_probability_context(
            CandidateFrame(_candidate_rows()),
            probabilities,
            fill_missing="error",
        )


def test_class_probability_context_cli_writes_outputs(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    probabilities = tmp_path / "probabilities.csv"
    output = tmp_path / "context_candidates.csv"
    provenance = tmp_path / "provenance.json"
    _candidate_rows().to_csv(candidates, index=False)
    _probability_rows().to_csv(probabilities, index=False)

    status = context_main(
        [
            "--candidate-csv",
            str(candidates),
            "--class-probabilities-csv",
            str(probabilities),
            "--output-csv",
            str(output),
            "--provenance-json",
            str(provenance),
        ]
    )

    assert status == 0
    rows = pd.read_csv(output)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert "image_class_prob_0" in rows.columns
    assert "image_class_prob_0_x_cluster_point_count" in rows.columns
    assert "image_class_prob_0_x_image_candidate_branch_dynamic_flag" in rows.columns
    assert payload["row_count"] == 3
    assert "image_class_prob_0" in payload["class_probability_columns"]
    assert "image_source_is_lidar_360" in payload["source_context_columns"]
