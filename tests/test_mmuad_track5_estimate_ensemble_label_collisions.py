from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import apply_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import load_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def _estimate(x_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )


def test_load_estimate_weight_config_rejects_normalized_label_collisions(
    tmp_path: Path,
) -> None:
    weights_json = tmp_path / "weights.json"
    weights_json.write_text(
        json.dumps({"weights": {"model a": 0.7, "model/a": 0.3}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ensemble weight labels collide after normalization"):
        load_estimate_weight_config(weights_json)


def test_apply_estimate_weight_config_rejects_colliding_runtime_labels() -> None:
    inputs = [
        EstimateInput("model a", Path("/tmp/a.csv"), 1.0),
        EstimateInput("model/a", Path("/tmp/b.csv"), 1.0),
    ]

    with pytest.raises(ValueError, match="estimate input labels collide after normalization"):
        apply_estimate_weight_config(inputs, {"model_a": 1.0}, missing_policy="keep")


def test_cli_parsed_estimate_specs_reject_duplicate_safe_labels() -> None:
    inputs = [
        parse_estimate_spec("model a=/tmp/a.csv"),
        parse_estimate_spec("model/a=/tmp/b.csv"),
    ]

    with pytest.raises(ValueError, match="estimate input label 'model_a' is duplicated"):
        apply_estimate_weight_config(inputs, {"model_a": 1.0}, missing_policy="keep")


def test_build_track5_estimate_ensemble_rejects_colliding_labels() -> None:
    with pytest.raises(ValueError, match="estimate input labels collide after normalization"):
        build_track5_estimate_ensemble(
            [
                ("model a", _estimate(0.0), 1.0),
                ("model/a", _estimate(1.0), 1.0),
            ],
            _template(),
        )
