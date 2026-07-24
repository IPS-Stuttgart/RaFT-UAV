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
    ("field", "value"),
    [
        ("measurement_sigma_m", np.ma.array(10.0, mask=True)),
        ("process_accel_std_mps2", np.ma.masked),
        ("initial_position_std_m", np.ma.array(100.0, mask=True)),
        ("initial_velocity_std_mps", np.ma.array(25.0, mask=True)),
        ("spread_variance_scale", np.ma.masked),
        ("max_nearest_time_delta_s", np.ma.masked),
    ],
)
def test_rts_ensemble_rejects_masked_controls_before_empty_return(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        build_track5_rts_ensemble(
            [("good", _estimate(), 1.0)],
            _template().iloc[0:0],
            **{field: value},
        )


def test_rts_ensemble_rejects_masked_runtime_weight() -> None:
    with pytest.raises(ValueError, match=r"weight\[bad\]"):
        build_track5_rts_ensemble(
            [("bad", _estimate(), np.ma.array(1.0, mask=True))],
            _template().iloc[0:0],
        )


def test_rts_writer_rejects_masked_weight_before_file_access(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=r"weight\[bad\]"):
        write_track5_rts_ensemble_outputs(
            estimate_inputs=[
                EstimateInput(
                    "bad",
                    tmp_path / "missing.csv",
                    np.ma.array(1.0, mask=True),
                )
            ],
            template=_template(),
            output_dir=output_dir,
        )

    assert not output_dir.exists()
