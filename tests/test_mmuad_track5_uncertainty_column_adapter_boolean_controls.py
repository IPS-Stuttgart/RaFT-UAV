from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_column_adapter import (
    normalize_uncertainty_estimate_inputs,
    write_uncertainty_column_adapter_outputs,
)


@pytest.mark.parametrize("value", ["False", "True", 0, 1, None, [], {}])
def test_uncertainty_adapter_rejects_non_boolean_require_uncertainty_before_io(
    tmp_path: Path,
    value: object,
) -> None:
    output_dir = tmp_path / "out"
    missing_csv = tmp_path / "missing.csv"

    with pytest.raises(ValueError, match="require_uncertainty must be a Boolean"):
        normalize_uncertainty_estimate_inputs(
            [EstimateInput("model", missing_csv, 1.0)],
            output_dir=output_dir,
            require_uncertainty=value,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("value", ["False", "True", 0, 1, None, [], {}])
def test_uncertainty_adapter_rejects_non_boolean_run_ensemble_before_io(
    tmp_path: Path,
    value: object,
) -> None:
    output_dir = tmp_path / "out"
    missing_csv = tmp_path / "missing.csv"

    with pytest.raises(ValueError, match="run_ensemble must be a Boolean"):
        write_uncertainty_column_adapter_outputs(
            estimate_inputs=[EstimateInput("model", missing_csv, 1.0)],
            output_dir=output_dir,
            run_ensemble=value,
        )

    assert not output_dir.exists()


def test_uncertainty_adapter_accepts_numpy_boolean_controls(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(estimate_csv, index=False)

    output_dir = tmp_path / "out"
    paths = write_uncertainty_column_adapter_outputs(
        estimate_inputs=[EstimateInput("model", estimate_csv, 1.0)],
        output_dir=output_dir,
        require_uncertainty=np.bool_(False),
        run_ensemble=np.bool_(False),
    )

    assert paths["summary_csv"].exists()
    normalized = pd.read_csv(output_dir / "normalized_estimates" / "model.csv")
    assert normalized["predicted_sigma_m"].tolist() == [30.0]
