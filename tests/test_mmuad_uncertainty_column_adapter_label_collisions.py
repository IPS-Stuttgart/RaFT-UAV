from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_column_adapter import _parse_uncertainty_column_map
from raft_uav.mmuad.track5_uncertainty_column_adapter import normalize_uncertainty_estimate_inputs


def _write_estimate(path: Path, *, x_m: float) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
            "predicted_sigma_m": [1.0],
        }
    ).to_csv(path, index=False)


def test_uncertainty_adapter_rejects_normalized_output_filename_collisions(
    tmp_path: Path,
) -> None:
    slash_csv = tmp_path / "slash.csv"
    space_csv = tmp_path / "space.csv"
    _write_estimate(slash_csv, x_m=1.0)
    _write_estimate(space_csv, x_m=99.0)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="estimate labels must be unique after normalization"):
        normalize_uncertainty_estimate_inputs(
            [
                EstimateInput("sensor/low", slash_csv, 1.0),
                EstimateInput("sensor low", space_csv, 1.0),
            ],
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_uncertainty_adapter_rejects_colliding_cli_column_labels() -> None:
    with pytest.raises(
        ValueError,
        match="uncertainty-column labels must be unique after normalization",
    ):
        _parse_uncertainty_column_map(
            [
                "sensor/low=predicted_sigma_m",
                "sensor low=state_sigma_m",
            ]
        )
