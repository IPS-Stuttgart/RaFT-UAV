from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_column_adapter import (
    normalize_uncertainty_estimate_inputs,
    write_uncertainty_column_adapter_outputs,
)


def _estimate(*, x_m: float, sigma_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
            "model_sigma": [sigma_m],
        }
    )


def test_uncertainty_adapter_rejects_input_output_alias_before_mutation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    normalized_dir = output_dir / "normalized_estimates"
    normalized_dir.mkdir(parents=True)
    later_input = normalized_dir / "first.csv"
    first_input = tmp_path / "first-source.csv"
    _estimate(x_m=99.0, sigma_m=9.0).to_csv(later_input, index=False)
    _estimate(x_m=1.0, sigma_m=1.0).to_csv(first_input, index=False)
    original_later_input = later_input.read_text(encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"normalized estimate outputs must not overwrite estimate inputs",
    ):
        normalize_uncertainty_estimate_inputs(
            [
                EstimateInput("first", first_input, 1.0),
                EstimateInput("later", later_input, 1.0),
            ],
            output_dir=output_dir,
            uncertainty_columns={"first": "model_sigma", "later": "model_sigma"},
        )

    assert later_input.read_text(encoding="utf-8") == original_later_input
    assert not (normalized_dir / "later.csv").exists()


def test_uncertainty_adapter_missing_template_fails_before_output_mutation(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    output_dir = tmp_path / "out"
    _estimate(x_m=1.0, sigma_m=1.0).to_csv(estimate_csv, index=False)

    with pytest.raises(ValueError, match="template is required when run_ensemble=True"):
        write_uncertainty_column_adapter_outputs(
            estimate_inputs=[EstimateInput("model", estimate_csv, 1.0)],
            output_dir=output_dir,
            uncertainty_columns={"model": "model_sigma"},
            run_ensemble=True,
        )

    assert not output_dir.exists()
