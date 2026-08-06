from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

ranker = importlib.import_module("raft_uav.mmuad.cluster_ranker")


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["lidar", "lidar", "radar", "radar"],
            "confidence": [0.9, 0.1, 0.8, 0.2],
            "good_cluster": [True, False, True, False],
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("learning_rate", -0.1),
        ("learning_rate", np.nan),
        ("learning_rate", True),
        ("learning_rate", np.asarray([0.05])),
        ("iterations", 0),
        ("iterations", 1.5),
        ("iterations", True),
        ("iterations", np.inf),
        ("iterations", np.asarray([10])),
        ("l2", -0.1),
        ("l2", np.nan),
        ("l2", True),
        ("l2", np.asarray([0.001])),
        ("random_state", -1),
        ("random_state", 1.5),
        ("random_state", True),
        ("random_state", np.inf),
        ("random_state", np.asarray([13])),
        ("random_state", np.iinfo(np.uint32).max + 1),
        ("n_estimators", 0),
        ("n_estimators", 1.5),
        ("n_estimators", True),
        ("n_estimators", np.inf),
        ("n_estimators", np.asarray([20])),
        ("score_distance_scale_m", 0.0),
        ("score_distance_scale_m", -1.0),
        ("score_distance_scale_m", np.inf),
        ("score_distance_scale_m", True),
        ("score_distance_scale_m", np.asarray([10.0])),
    ],
)
def test_cluster_ranker_rejects_invalid_training_control(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        ranker.train_cluster_ranker(_features(), **{field: value})


def test_cluster_ranker_normalizes_valid_scalar_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def _capture(features: pd.DataFrame, **kwargs: Any) -> object:
        captured["features"] = features
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(ranker, "_LEGACY_TRAIN_CLUSTER_RANKER", _capture)

    result = ranker.train_cluster_ranker(
        _features(),
        learning_rate="0.05",
        iterations="12",
        l2=np.asarray(0.001),
        random_state=np.int64(7),
        n_estimators="25",
        score_distance_scale_m=np.float64(8.0),
    )

    assert result is sentinel
    assert captured["learning_rate"] == pytest.approx(0.05)
    assert captured["iterations"] == 12
    assert type(captured["iterations"]) is int
    assert captured["l2"] == pytest.approx(0.001)
    assert captured["random_state"] == 7
    assert type(captured["random_state"]) is int
    assert captured["n_estimators"] == 25
    assert type(captured["n_estimators"]) is int
    assert captured["score_distance_scale_m"] == pytest.approx(8.0)
    assert captured["features"]["good_cluster"].dtype.name == "boolean"
