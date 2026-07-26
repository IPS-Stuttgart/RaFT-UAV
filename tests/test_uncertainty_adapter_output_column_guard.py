from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_column_adapter import normalize_uncertainty_estimate_inputs


def _estimate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [10.0, 11.0],
            "state_y_m": [20.0, 21.0],
            "state_z_m": [30.0, 31.0],
            "model_sigma": [2.0, 3.0],
        }
    )


@pytest.mark.parametrize(
    "output_column",
    [
        None,
        "",
        " state_x_m ",
        "STATE_Y_M",
        "Sequence",
        "Timestamp",
        "Position",
        "Classification",
        "x",
        "east_m",
    ],
)
def test_uncertainty_adapter_rejects_trajectory_output_columns(
    tmp_path: Path,
    output_column: str | None,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    output_dir = tmp_path / "out"
    _estimate_rows().to_csv(estimate_csv, index=False)

    with pytest.raises(ValueError, match="trajectory-defining|must not be empty"):
        normalize_uncertainty_estimate_inputs(
            [EstimateInput("model", estimate_csv, 1.0)],
            output_dir=output_dir,
            uncertainty_columns={"model": "model_sigma"},
            output_uncertainty_column=output_column,  # type: ignore[arg-type]
        )

    assert not output_dir.exists()


def test_uncertainty_adapter_preserves_trajectory_for_custom_output_column(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate_rows().to_csv(estimate_csv, index=False)

    normalized, _ = normalize_uncertainty_estimate_inputs(
        [EstimateInput("model", estimate_csv, 1.0)],
        output_dir=tmp_path / "out",
        uncertainty_columns={"model": "model_sigma"},
        output_uncertainty_column="custom_sigma_m",
    )

    rows = pd.read_csv(normalized[0].path)
    assert rows["state_x_m"].tolist() == [10.0, 11.0]
    assert rows["state_y_m"].tolist() == [20.0, 21.0]
    assert rows["state_z_m"].tolist() == [30.0, 31.0]
    assert rows["custom_sigma_m"].tolist() == [2.0, 3.0]
