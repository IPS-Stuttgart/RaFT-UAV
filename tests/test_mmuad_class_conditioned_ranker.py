from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.class_conditioned_ranker import main as ranker_main


def _candidate_rows() -> pd.DataFrame:
    records = []
    for sequence_id in ("seqA", "seqB"):
        records.extend(
            [
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "lidar_360",
                    "track_id": f"{sequence_id}-good",
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "confidence": 0.5,
                    "cluster_point_count": 20,
                    "cluster_extent_3d_m": 1.0,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "lidar_360",
                    "track_id": f"{sequence_id}-bad",
                    "x_m": 10.0,
                    "y_m": 10.0,
                    "z_m": 4.0,
                    "confidence": 0.8,
                    "cluster_point_count": 5,
                    "cluster_extent_3d_m": 5.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )


def _class_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "predicted_probability_0": [0.9, 0.1],
            "predicted_probability_1": [0.05, 0.8],
            "predicted_probability_2": [0.03, 0.05],
            "predicted_probability_3": [0.02, 0.05],
            "predicted_class": [0, 1],
        }
    )


def test_class_conditioned_ranker_trains_and_scores(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    probabilities_csv = tmp_path / "probabilities.csv"
    model_json = tmp_path / "model.json"
    scored_csv = tmp_path / "scored.csv"
    train_features_csv = tmp_path / "train_features.csv"
    score_features_csv = tmp_path / "score_features.csv"
    train_context_csv = tmp_path / "train_context.csv"
    score_context_csv = tmp_path / "score_context.csv"
    provenance_json = tmp_path / "provenance.json"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)
    _class_probabilities().to_csv(probabilities_csv, index=False)

    status = ranker_main(
        [
            "--train-candidates",
            str(candidates_csv),
            "--train-truth",
            str(truth_csv),
            "--train-class-probabilities-csv",
            str(probabilities_csv),
            "--score-candidates",
            str(candidates_csv),
            "--score-class-probabilities-csv",
            str(probabilities_csv),
            "--model-json",
            str(model_json),
            "--scored-candidates-csv",
            str(scored_csv),
            "--train-features-csv",
            str(train_features_csv),
            "--score-features-csv",
            str(score_features_csv),
            "--train-context-candidates-csv",
            str(train_context_csv),
            "--score-context-candidates-csv",
            str(score_context_csv),
            "--provenance-json",
            str(provenance_json),
            "--good-threshold-m",
            "1.0",
            "--max-truth-time-delta-s",
            "0.1",
            "--iterations",
            "40",
            "--interaction-column",
            "cluster_point_count",
            "--interaction-column",
            "cluster_extent_3d_m",
        ]
    )

    assert status == 0
    model = json.loads(model_json.read_text(encoding="utf-8"))
    scored = pd.read_csv(scored_csv)
    provenance = json.loads(provenance_json.read_text(encoding="utf-8"))
    assert "image_class_prob_0" in model["feature_columns"]
    assert "image_class_prob_0_x_cluster_point_count" in model["feature_columns"]
    assert "ranker_score" in scored.columns
    assert "image_class_entropy" in scored.columns
    assert scored["ranker_score"].notna().all()
    assert provenance["protocol"] == (
        "soft sequence-class probabilities as candidate-ranker context"
    )
    assert "image_class_prob_1" in provenance["class_context_feature_columns"]
    for path in (
        train_features_csv,
        score_features_csv,
        train_context_csv,
        score_context_csv,
    ):
        assert path.exists()


def test_class_conditioned_ranker_requires_train_probabilities(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    with pytest.raises(SystemExit, match="train-class-probabilities"):
        ranker_main(
            [
                "--train-candidates",
                str(candidates_csv),
                "--train-truth",
                str(truth_csv),
                "--model-json",
                str(tmp_path / "model.json"),
            ]
        )


def test_class_conditioned_ranker_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["raft-uav-mmuad-class-conditioned-ranker"] == (
        "raft_uav.mmuad.class_conditioned_ranker:main"
    )
