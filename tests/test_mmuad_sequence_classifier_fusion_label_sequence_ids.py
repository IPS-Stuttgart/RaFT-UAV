from __future__ import annotations

import pandas as pd
import pytest

import raft_uav.mmuad.sequence_classifier_fusion as fusion


def _one_hot_probabilities(sequence_id: str, class_index: int) -> pd.DataFrame:
    row: dict[str, object] = {"sequence_id": sequence_id}
    for index, label in enumerate(fusion.OFFICIAL_SEQUENCE_CLASS_LABELS):
        row[f"predicted_probability_{label}"] = 1.0 if index == class_index else 0.0
    return pd.DataFrame([row])


def test_probability_fusion_normalizes_eval_label_sequence_keys() -> None:
    fused = fusion.fuse_sequence_probabilities(
        _one_hot_probabilities(" seq-a ", 0),
        _one_hot_probabilities("seq-a", 0),
        image_weight=0.5,
        eval_labels={" seq-a ": "0"},
    )

    assert fused["sequence_id"].tolist() == ["seq-a"]
    assert fused["ground_truth_class"].tolist() == ["0"]
    assert fused["correct"].tolist() == [True]


def test_label_map_rejects_collisions_after_sequence_key_normalization() -> None:
    with pytest.raises(
        ValueError,
        match="eval_labels contains duplicate sequence_id keys after normalization: 'seq-a'",
    ):
        fusion._validated_label_map(
            {"seq-a": "0", " seq-a ": "1"},
            name="eval_labels",
        )


def test_label_map_rejects_missing_sequence_keys() -> None:
    with pytest.raises(ValueError, match="train_labels contains missing sequence_id keys"):
        fusion._validated_label_map({None: "0"}, name="train_labels")


def test_selection_normalizes_train_and_eval_label_keys_before_legacy(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_select(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(fusion, "_LEGACY_SELECT_TRAIN_SAFE_FUSION", fake_select)

    result = fusion.select_train_safe_fusion(
        image_train_features=pd.DataFrame(),
        nonimage_train_features=pd.DataFrame(),
        image_predict_features=pd.DataFrame(),
        nonimage_predict_features=pd.DataFrame(),
        train_labels={" seq-a ": "0", "001": "1"},
        eval_labels={" seq-b ": "2"},
        model_specs=[],
        image_weights=[0.5],
    )

    assert result is sentinel
    assert captured["train_labels"] == {"seq-a": "0", "001": "1"}
    assert captured["eval_labels"] == {"seq-b": "2"}
