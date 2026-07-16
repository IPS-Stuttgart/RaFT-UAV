from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_rts_ensemble import build_track5_rts_ensemble
from raft_uav.mmuad.track5_rts_ensemble import write_track5_rts_ensemble_outputs


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


@pytest.mark.parametrize(
    "weight",
    [
        True,
        False,
        np.bool_(True),
        np.asarray(True),
        np.asarray([1.0]),
    ],
)
def test_rts_ensemble_rejects_non_numeric_scalar_weights_before_empty_return(
    weight: object,
) -> None:
    with pytest.raises(ValueError, match=r"weight\[bad\] must be positive and finite"):
        build_track5_rts_ensemble(
            [("bad", _estimate(), weight)],
            _template().iloc[0:0],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measurement_sigma_m", True, "measurement_sigma_m must be positive and finite"),
        (
            "process_accel_std_mps2",
            np.asarray([1.0]),
            "process_accel_std_mps2 must be non-negative and finite",
        ),
        (
            "initial_position_std_m",
            1.0 + 0.0j,
            "initial_position_std_m must be positive and finite",
        ),
        (
            "initial_velocity_std_mps",
            np.nan,
            "initial_velocity_std_mps must be positive and finite",
        ),
        (
            "spread_variance_scale",
            np.bool_(False),
            "spread_variance_scale must be non-negative and finite",
        ),
        (
            "max_nearest_time_delta_s",
            True,
            "max_nearest_time_delta_s must be non-negative and finite",
        ),
    ],
)
def test_rts_ensemble_rejects_invalid_controls_before_empty_return(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_track5_rts_ensemble(
            [("good", _estimate(), 1.0)],
            _template().iloc[0:0],
            **{field: value},
        )


def test_rts_writer_validates_weight_before_missing_file_access(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=r"weight\[bad\] must be positive and finite"):
        write_track5_rts_ensemble_outputs(
            estimate_inputs=[EstimateInput("bad", tmp_path / "missing.csv", True)],
            template=_template(),
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_rts_ensemble_accepts_real_numpy_scalars() -> None:
    estimates, diagnostics = build_track5_rts_ensemble(
        [("good", _estimate(), np.float64(1.0))],
        _template(),
        measurement_sigma_m=np.asarray(1.0),
        process_accel_std_mps2=np.float64(0.1),
        initial_position_std_m=np.int64(10),
        initial_velocity_std_mps=np.float64(2.0),
        spread_variance_scale=np.int64(0),
        max_nearest_time_delta_s=np.asarray(0.0),
    )

    assert estimates[["state_x_m", "state_y_m", "state_z_m"]].iloc[0].tolist() == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert diagnostics["valid_input_count"].tolist() == [1]
