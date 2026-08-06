from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_column_adapter import (
    normalize_uncertainty_estimate_inputs,
)


def _write_estimate(path: Path) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "model_sigma": [2.5],
        }
    ).to_csv(path, index=False)


def test_uncertainty_adapter_rejects_unused_mapping_labels(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _write_estimate(estimate_csv)

    with pytest.raises(ValueError, match="do not match estimate inputs"):
        normalize_uncertainty_estimate_inputs(
            [EstimateInput("model", estimate_csv, 1.0)],
            output_dir=tmp_path / "out",
            uncertainty_columns={"modle": "model_sigma"},
        )


def test_uncertainty_adapter_rejects_two_aliases_for_one_estimate(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _write_estimate(estimate_csv)

    with pytest.raises(ValueError, match="at most one mapping per estimate"):
        normalize_uncertainty_estimate_inputs(
            [EstimateInput("sensor/model", estimate_csv, 1.0)],
            output_dir=tmp_path / "out",
            uncertainty_columns={
                "sensor/model": "model_sigma",
                "sensor_model": "model_sigma",
            },
        )
