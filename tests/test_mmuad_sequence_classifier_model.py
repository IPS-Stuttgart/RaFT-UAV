from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.classification import load_sequence_classifier_model
from raft_uav.mmuad.run import main as mmuad_run_main
from raft_uav.mmuad.train_sequence_classifier import main as train_sequence_classifier_main


def _write_sequence(root: Path, sequence_id: str, xyz: tuple[float, float, float]) -> None:
    sequence = root / sequence_id
    sequence.mkdir(parents=True, exist_ok=True)
    x_m, y_m, z_m = xyz
    pd.DataFrame(
        {
            "sequence_id": [sequence_id, sequence_id],
            "time_s": [0.0, 1.0],
            "source": ["lidar_360", "lidar_360"],
            "x_m": [x_m, x_m + 0.1],
            "y_m": [y_m, y_m + 0.1],
            "z_m": [z_m, z_m],
        }
    ).to_csv(sequence / "candidates.csv", index=False)


def test_train_sequence_classifier_and_copy_prediction_to_submission(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    _write_sequence(train_root, "seq_train_0", (0.0, 0.0, 10.0))
    _write_sequence(train_root, "seq_train_3", (100.0, 100.0, 30.0))
    _write_sequence(val_root, "seq_target", (100.2, 100.1, 30.0))
    reference = tmp_path / "train_reference.csv"
    pd.DataFrame(
        {"sequence_id": ["seq_train_0", "seq_train_3"], "uav_type": [0, 3]}
    ).to_csv(reference, index=False)
    model_path = tmp_path / "outputs" / "mmuad_sequence_classifier_rf.joblib"
    feature_report = tmp_path / "outputs" / "features.csv"

    train_status = train_sequence_classifier_main(
        [
            str(train_root),
            "--reference",
            str(reference),
            "--method",
            "random-forest",
            "--output",
            str(model_path),
            "--feature-report",
            str(feature_report),
            "--n-estimators",
            "30",
        ]
    )

    assert train_status == 0
    assert model_path.exists()
    assert feature_report.exists()
    model = load_sequence_classifier_model(model_path)
    assert model["method"] == "random-forest"
    assert model["prediction_mode"] == "sequence_level"
    assert set(model["train_sequences"]) == {"seq_train_0", "seq_train_3"}

    output = tmp_path / "run"
    results = output / "mmaud_results.csv"
    run_status = mmuad_run_main(
        [
            str(val_root),
            "--sequence-classifier",
            str(model_path),
            "--output-dir",
            str(output),
            "--ug2-official-results-csv",
            str(results),
        ]
    )

    assert run_status == 0
    rows = pd.read_csv(results)
    assert rows["Sequence"].tolist() == ["seq_target", "seq_target"]
    assert rows["Classification"].tolist() == [3, 3]
    predictions = pd.read_csv(output / "mmuad_sequence_classifier_predictions.csv")
    assert predictions[["sequence_id", "predicted_class"]].to_dict("records") == [
        {"sequence_id": "seq_target", "predicted_class": 3}
    ]
    provenance = json.loads(
        (output / "mmuad_sequence_classifier_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["classification_model_path"] == str(model_path)
    assert provenance["classification_method"] == "random-forest"
    assert provenance["classification_prediction_mode"] == "sequence_level"
    assert provenance["classification_class_map"] == {"seq_target": "3"}
