from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.classification_cli import main as sequence_classifier_main


def test_sequence_classifier_cli_accepts_method_aliases(tmp_path: Path) -> None:
    train_path = tmp_path / "train_features.csv"
    predict_path = tmp_path / "predict_features.csv"
    labels_path = tmp_path / "labels.csv"
    class_map_path = tmp_path / "class_map.csv"
    metrics_path = tmp_path / "metrics.json"

    pd.DataFrame(
        {
            "sequence_id": ["seq0", "seq1"],
            "cluster_point_count_mean": [1.0, 10.0],
            "cluster_extent_3d_m_mean": [0.5, 3.0],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["target"],
            "cluster_point_count_mean": [9.5],
            "cluster_extent_3d_m_mean": [2.8],
        }
    ).to_csv(predict_path, index=False)
    pd.DataFrame({"sequence_id": ["seq0", "seq1"], "uav_type": [0, 1]}).to_csv(
        labels_path,
        index=False,
    )

    status = sequence_classifier_main(
        [
            "--train-feature-table",
            str(train_path),
            "--predict-feature-table",
            str(predict_path),
            "--train-labels",
            str(labels_path),
            "--method",
            "NN",
            "--output-class-map",
            str(class_map_path),
            "--metrics-json",
            str(metrics_path),
        ]
    )

    assert status == 0
    assert pd.read_csv(class_map_path).to_dict("records") == [
        {"sequence_id": "target", "uav_type": 1}
    ]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["method"] == "nearest-neighbor"
