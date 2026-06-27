from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.sequence_classifier_fusion import (
    FusionModelSpec,
    SELECTED_PROBABILITIES_CSV,
    main as fusion_main,
    select_train_safe_fusion,
)


def test_train_safe_fusion_selects_weight_from_train_cv_only(tmp_path: Path) -> None:
    image_train, nonimage_train, image_predict, nonimage_predict, labels, eval_labels = (
        _fusion_fixture()
    )

    result = select_train_safe_fusion(
        image_train_features=image_train,
        nonimage_train_features=nonimage_train,
        image_predict_features=image_predict,
        nonimage_predict_features=nonimage_predict,
        train_labels=labels,
        eval_labels=eval_labels,
        model_specs=[
            FusionModelSpec(
                method="nearest-neighbor",
                n_estimators=1,
                max_depth=None,
                random_state=13,
            )
        ],
        image_weights=[0.0, 1.0],
        cv_folds=2,
    )

    assert result.manifest["selected"]["image_weight"] == 0.0
    assert result.selected_probabilities["predicted_class"].astype(str).tolist() == ["0", "1"]
    assert result.manifest["selected_predict_accuracy_diagnostic"] == 1.0


def test_sequence_classifier_fusion_cli_writes_train_selected_predictions(
    tmp_path: Path,
) -> None:
    image_train, nonimage_train, image_predict, nonimage_predict, labels, eval_labels = (
        _fusion_fixture()
    )
    image_train_csv = tmp_path / "image_train.csv"
    nonimage_train_csv = tmp_path / "nonimage_train.csv"
    image_predict_csv = tmp_path / "image_predict.csv"
    nonimage_predict_csv = tmp_path / "nonimage_predict.csv"
    train_labels_csv = tmp_path / "train_labels.csv"
    eval_labels_csv = tmp_path / "eval_labels.csv"
    image_train.to_csv(image_train_csv, index=False)
    nonimage_train.to_csv(nonimage_train_csv, index=False)
    image_predict.to_csv(image_predict_csv, index=False)
    nonimage_predict.to_csv(nonimage_predict_csv, index=False)
    pd.DataFrame(
        {"sequence_id": list(labels.keys()), "uav_type": list(labels.values())}
    ).to_csv(train_labels_csv, index=False)
    pd.DataFrame(
        {"sequence_id": list(eval_labels.keys()), "uav_type": list(eval_labels.values())}
    ).to_csv(eval_labels_csv, index=False)
    output_dir = tmp_path / "out"

    status = fusion_main(
        [
            "--image-train-features",
            str(image_train_csv),
            "--nonimage-train-features",
            str(nonimage_train_csv),
            "--image-predict-features",
            str(image_predict_csv),
            "--nonimage-predict-features",
            str(nonimage_predict_csv),
            "--train-labels",
            str(train_labels_csv),
            "--eval-labels",
            str(eval_labels_csv),
            "--output-dir",
            str(output_dir),
            "--method",
            "nearest-neighbor",
            "--image-weight-grid",
            "0,1",
            "--cv-folds",
            "2",
        ]
    )

    assert status == 0
    manifest = json.loads(
        (output_dir / "mmuad_train_safe_fusion_weight_probe.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["selection_protocol"].startswith("Stratified train-label CV")
    assert manifest["selected"]["image_weight"] == 0.0
    selected = pd.read_csv(output_dir / SELECTED_PROBABILITIES_CSV)
    assert selected["predicted_class"].astype(str).tolist() == ["0", "1"]
    assert selected["correct"].tolist() == [True, True]
    diagnostics = pd.read_csv(
        output_dir / "mmuad_train_safe_fusion_weight_predict_diagnostic.csv"
    )
    assert diagnostics.loc[
        diagnostics["image_weight"].eq(0.0), "predict_accuracy_diagnostic"
    ].iloc[0] == 1.0


def _fusion_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    dict[str, str],
]:
    train_sequences = ["seq0a", "seq0b", "seq0c", "seq1a", "seq1b", "seq1c"]
    labels = {
        "seq0a": "0",
        "seq0b": "0",
        "seq0c": "0",
        "seq1a": "1",
        "seq1b": "1",
        "seq1c": "1",
    }
    nonimage_values = [0.0, 0.1, 0.2, 10.0, 9.9, 9.8]
    image_values = [10.0, 9.9, 9.8, 0.0, 0.1, 0.2]
    image_train = pd.DataFrame(
        {"sequence_id": train_sequences, "image_signal": image_values}
    )
    nonimage_train = pd.DataFrame(
        {"sequence_id": train_sequences, "nonimage_signal": nonimage_values}
    )
    image_predict = pd.DataFrame(
        {"sequence_id": ["seqVal0", "seqVal1"], "image_signal": [9.85, 0.15]}
    )
    nonimage_predict = pd.DataFrame(
        {"sequence_id": ["seqVal0", "seqVal1"], "nonimage_signal": [0.15, 9.85]}
    )
    eval_labels = {"seqVal0": "0", "seqVal1": "1"}
    return image_train, nonimage_train, image_predict, nonimage_predict, labels, eval_labels
