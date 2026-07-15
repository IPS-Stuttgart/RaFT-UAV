from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import apply_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import load_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import (
    write_track5_estimate_ensemble_outputs,
)
from raft_uav.mmuad.track5_uncertainty_ensemble import (
    build_track5_uncertainty_ensemble,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def _estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "predicted_sigma_m": [1.0],
        }
    )


@pytest.mark.parametrize(
    "weight",
    [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.array(True),
        np.array([False]),
    ],
)
def test_estimate_ensemble_rejects_boolean_runtime_weights(weight: object) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_track5_estimate_ensemble(
            [("bad", _estimate(), weight)],
            _template(),
        )


@pytest.mark.parametrize(
    "weight",
    [True, False, np.bool_(True), np.bool_(False), np.array(True)],
)
def test_estimate_ensemble_validates_weights_before_empty_template_return(
    weight: object,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_track5_estimate_ensemble(
            [("bad", _estimate(), weight)],
            _template().iloc[0:0],
        )


@pytest.mark.parametrize("weight", [True, False])
def test_weight_config_rejects_json_booleans(tmp_path: Path, weight: bool) -> None:
    config = tmp_path / "weights.json"
    config.write_text(
        json.dumps({"weights": {"bad": weight}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        load_estimate_weight_config(config)


def test_keep_policy_rejects_boolean_inline_weight(tmp_path: Path) -> None:
    inputs = [EstimateInput("bad", tmp_path / "missing.csv", True)]

    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_estimate_weight_config(inputs, {}, missing_policy="keep")


def test_writer_validates_weight_before_estimate_file_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        write_track5_estimate_ensemble_outputs(
            estimate_inputs=[EstimateInput("bad", tmp_path / "missing.csv", True)],
            template=_template(),
            output_dir=tmp_path / "out",
        )


@pytest.mark.parametrize("weight", [True, np.bool_(False)])
def test_uncertainty_ensemble_rejects_boolean_global_weights(
    tmp_path: Path,
    weight: object,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_track5_uncertainty_ensemble(
            [EstimateInput("bad", tmp_path / "missing.csv", weight)],
            template=_template().iloc[0:0],
        )


def test_ordinary_numeric_zero_and_one_weights_remain_valid() -> None:
    ensemble, diagnostics = build_track5_estimate_ensemble(
        [
            ("disabled", _estimate().assign(state_x_m=100.0), 0),
            ("enabled", _estimate(), np.int64(1)),
        ],
        _template(),
    )

    assert ensemble["state_x_m"].tolist() == pytest.approx([1.0])
    assert diagnostics["valid_input_count"].tolist() == [1]
