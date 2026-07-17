from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import apply_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import (
    write_track5_estimate_ensemble_outputs,
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
        }
    )


@pytest.mark.parametrize("weight", [np.array([0.5]), np.array([[0.5]])])
def test_estimate_ensemble_rejects_non_scalar_weights(weight: object) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_track5_estimate_ensemble(
            [("bad", _estimate(), weight)],
            _template(),
        )


def test_weight_config_rejects_non_scalar_programmatic_weights(tmp_path: Path) -> None:
    inputs = [EstimateInput("candidate", tmp_path / "missing.csv", 1.0)]

    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_estimate_weight_config(
            inputs,
            {"candidate": np.array([0.5])},
        )


@pytest.mark.parametrize(
    "trim_fraction",
    [
        False,
        np.bool_(False),
        np.array(False),
        np.array([0.25]),
        0.25 + 0.0j,
        np.ma.masked,
    ],
)
def test_estimate_ensemble_rejects_malformed_trim_fractions(
    trim_fraction: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"trim_fraction must be a finite real scalar in \[0, 0.5\)",
    ):
        build_track5_estimate_ensemble(
            [("candidate", _estimate(), 1.0)],
            _template(),
            aggregation_policy="trimmed-mean",
            trim_fraction=trim_fraction,
        )


def test_estimate_ensemble_accepts_zero_dimensional_real_controls() -> None:
    ensemble, diagnostics = build_track5_estimate_ensemble(
        [("candidate", _estimate(), np.array(0.5))],
        _template(),
        aggregation_policy="trimmed-mean",
        trim_fraction=np.array(0.25),
    )

    assert ensemble["state_x_m"].tolist() == pytest.approx([1.0])
    assert ensemble["ensemble_weight_sum"].tolist() == pytest.approx([0.5])
    assert diagnostics["valid_input_count"].tolist() == [1]


def test_writer_validates_trim_fraction_before_file_access(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="trim_fraction"):
        write_track5_estimate_ensemble_outputs(
            estimate_inputs=[
                EstimateInput("candidate", tmp_path / "missing.csv", 1.0)
            ],
            template=_template(),
            output_dir=output_dir,
            trim_fraction=np.array([0.25]),
        )

    assert not output_dir.exists()
