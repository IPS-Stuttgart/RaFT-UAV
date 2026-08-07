from __future__ import annotations

import pandas as pd

import raft_uav.mmuad.sequence_classifier_fusion as fusion


def test_colliding_fusion_specs_keep_the_selected_random_state(monkeypatch) -> None:
    model_specs = [
        fusion.FusionModelSpec(
            method="nearest-neighbor",
            n_estimators=1,
            max_depth=None,
            random_state=13,
        ),
        fusion.FusionModelSpec(
            method="nearest-neighbor",
            n_estimators=1,
            max_depth=None,
            random_state=29,
        ),
    ]
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_select_train_safe_fusion(**kwargs: object) -> object:
        delegated_specs = kwargs["model_specs"]
        assert isinstance(delegated_specs, list)
        selected = delegated_specs[1]
        captured["delegated_specs"] = delegated_specs
        captured["matched_spec"] = fusion._IMPL._matching_spec(
            delegated_specs,
            {"model_name": selected.name},
        )
        return sentinel

    monkeypatch.setattr(
        fusion,
        "_LEGACY_SELECT_TRAIN_SAFE_FUSION",
        fake_select_train_safe_fusion,
    )

    result = fusion.select_train_safe_fusion(
        image_train_features=pd.DataFrame(),
        nonimage_train_features=pd.DataFrame(),
        image_predict_features=pd.DataFrame(),
        nonimage_predict_features=pd.DataFrame(),
        train_labels={},
        model_specs=model_specs,
        image_weights=[0.5],
    )

    assert result is sentinel
    delegated_specs = captured["delegated_specs"]
    assert isinstance(delegated_specs, list)
    assert delegated_specs[0].name != delegated_specs[1].name
    assert [spec.random_state for spec in delegated_specs] == [13, 29]
    assert captured["matched_spec"].random_state == 29
