from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mmuad_class_context_ranker_train_val.py"
spec = importlib.util.spec_from_file_location("mmuad_class_context_ranker_train_val", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _candidate_rows(sequence_id: str = "seqA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [sequence_id, sequence_id],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "livox_avia"],
            "track_id": ["good", "bad"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 10.0],
            "z_m": [1.0, 4.0],
            "confidence": [1.0, 0.5],
            "cluster_point_count": [20, 4],
            "cluster_extent_3d_m": [1.0, 5.0],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )


def _probability_rows(sequence_id: str = "seqA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [sequence_id],
            "predicted_probability_0": [0.05],
            "predicted_probability_1": [0.10],
            "predicted_probability_2": [0.80],
            "predicted_probability_3": [0.05],
            "predicted_class": [2],
        }
    )


def test_class_context_ranker_runner_trains_scores_and_writes_context_features(tmp_path: Path) -> None:
    train_candidates = tmp_path / "train_candidates.csv"
    score_candidates = tmp_path / "score_candidates.csv"
    truth = tmp_path / "truth.csv"
    train_probs = tmp_path / "train_probs.csv"
    score_probs = tmp_path / "score_probs.csv"
    model_json = tmp_path / "model.json"
    scored_csv = tmp_path / "scored.csv"
    train_context_csv = tmp_path / "train_context.csv"
    train_features_csv = tmp_path / "train_features.csv"
    score_features_csv = tmp_path / "score_features.csv"
    summary_json = tmp_path / "summary.json"

    _candidate_rows().to_csv(train_candidates, index=False)
    _candidate_rows().to_csv(score_candidates, index=False)
    _truth_rows().to_csv(truth, index=False)
    _probability_rows().to_csv(train_probs, index=False)
    _probability_rows().to_csv(score_probs, index=False)

    status = runner.main(
        [
            "--train-candidates",
            str(train_candidates),
            "--score-candidates",
            str(score_candidates),
            "--train-truth",
            str(truth),
            "--train-class-probabilities-csv",
            str(train_probs),
            "--score-class-probabilities-csv",
            str(score_probs),
            "--model-json",
            str(model_json),
            "--scored-candidates-csv",
            str(scored_csv),
            "--train-context-candidates-csv",
            str(train_context_csv),
            "--train-features-csv",
            str(train_features_csv),
            "--score-features-csv",
            str(score_features_csv),
            "--summary-json",
            str(summary_json),
            "--model-type",
            "logistic",
            "--target-column",
            "good_cluster_5m",
            "--good-threshold-m",
            "5.0",
            "--max-truth-time-delta-s",
            "0.1",
            "--interaction-column",
            "cluster_point_count",
        ]
    )

    assert status == 0
    assert model_json.exists()
    assert scored_csv.exists()
    context_rows = pd.read_csv(train_context_csv)
    train_features = pd.read_csv(train_features_csv)
    score_features = pd.read_csv(score_features_csv)
    scored = pd.read_csv(scored_csv).sort_values("x_m")
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    assert "image_class_prob_2" in context_rows.columns
    assert "image_class_prob_2_x_cluster_point_count" in train_features.columns
    assert "image_class_prob_2_x_cluster_point_count" in score_features.columns
    assert summary["class_context_feature_count"] > 0
    assert scored["ranker_score"].notna().all()
    assert scored["ranker_score"].iloc[0] > scored["ranker_score"].iloc[1]
