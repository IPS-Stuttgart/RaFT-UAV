from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_column_adapter import (
    normalize_uncertainty_estimate_inputs,
)


def _estimate_with_uncertainty_columns(
    columns: list[str],
    values: list[float],
) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    )
    for column, value in zip(columns, values, strict=True):
        rows.insert(len(rows.columns), column, [value], allow_duplicates=True)
    return rows


def test_uncertainty_adapter_rejects_ambiguous_requested_column(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate_with_uncertainty_columns(
        ["model_sigma", " MODEL_SIGMA "],
        [2.0, 20.0],
    ).to_csv(estimate_csv, index=False)

    with pytest.raises(ValueError, match="ambiguous uncertainty columns"):
        normalize_uncertainty_estimate_inputs(
            [EstimateInput("model", estimate_csv, 1.0)],
            output_dir=tmp_path / "out",
            uncertainty_columns={"model": "model_sigma"},
        )


def test_uncertainty_adapter_rejects_exact_duplicate_default_column(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate_with_uncertainty_columns(
        ["predicted_sigma_m", "predicted_sigma_m"],
        [2.0, 20.0],
    ).to_csv(estimate_csv, index=False)

    with pytest.raises(ValueError, match="ambiguous uncertainty columns"):
        normalize_uncertainty_estimate_inputs(
            [EstimateInput("model", estimate_csv, 1.0)],
            output_dir=tmp_path / "out",
        )


def test_uncertainty_adapter_ignores_unrelated_normalized_collisions(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate_with_uncertainty_columns(
        ["diagnostic", " DIAGNOSTIC ", "model_sigma"],
        [1.0, 2.0, 3.0],
    ).to_csv(estimate_csv, index=False)

    normalized, summary = normalize_uncertainty_estimate_inputs(
        [EstimateInput("model", estimate_csv, 1.0)],
        output_dir=tmp_path / "out",
        uncertainty_columns={"model": "model_sigma"},
    )

    rows = pd.read_csv(normalized[0].path)
    assert rows["predicted_sigma_m"].tolist() == [3.0]
    assert summary["source_uncertainty_column"].tolist() == ["model_sigma"]
