from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import apply_ensemble_weight_config
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import load_ensemble_weight_config
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import write_apply_weights_outputs


@pytest.mark.parametrize("payload", [None, [], "weights", 1])
def test_weight_config_rejects_non_object_json(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="weight config must be a JSON object"):
        load_ensemble_weight_config(path)


@pytest.mark.parametrize("weight", [True, False])
def test_weight_config_rejects_json_boolean_weights(tmp_path: Path, weight: bool) -> None:
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"weights": {"a": weight}}), encoding="utf-8")

    with pytest.raises(ValueError, match="finite and non-negative"):
        load_ensemble_weight_config(path)


@pytest.mark.parametrize(
    "default_weight",
    [True, np.bool_(False), np.array(True), np.array([1.0])],
)
def test_default_missing_weight_rejects_non_real_scalars(
    tmp_path: Path,
    default_weight: object,
) -> None:
    inputs = [
        EstimateInput("a", tmp_path / "a.csv", 1.0),
        EstimateInput("b", tmp_path / "b.csv", 1.0),
    ]

    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_ensemble_weight_config(
            inputs,
            {"weights": {"a": 1.0}},
            missing_weight_policy="default",
            default_missing_weight=default_weight,
        )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trim_fraction", True, "trim_fraction"),
        ("max_nearest_time_delta_s", np.bool_(False), "max_nearest_time_delta_s"),
    ],
)
def test_apply_weight_output_rejects_boolean_numeric_controls(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    template_path = tmp_path / "template.csv"
    _template().to_csv(template_path, index=False)
    weight_config = {"weights": {"a": 1.0}, field: value}

    with pytest.raises(ValueError, match=message):
        write_apply_weights_outputs(
            estimate_inputs=[EstimateInput("a", tmp_path / "unused.csv", 1.0)],
            weight_config=weight_config,
            template_path=template_path,
            output_dir=tmp_path / "out",
        )
