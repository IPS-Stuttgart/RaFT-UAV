from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_rts_ensemble import write_track5_rts_ensemble_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})


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


def test_rts_writer_rejects_empty_inputs_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="at least one estimate input"):
        write_track5_rts_ensemble_outputs(
            estimate_inputs=[],
            template=_template(),
            output_dir=output,
        )

    assert not output.exists()


def test_rts_writer_validates_parameters_before_creating_output(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate().to_csv(estimate_csv, index=False)
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="measurement_sigma_m must be positive and finite"):
        write_track5_rts_ensemble_outputs(
            estimate_inputs=[parse_estimate_spec(f"estimate={estimate_csv}@1.0")],
            template=_template(),
            output_dir=output,
            measurement_sigma_m=0.0,
        )

    assert not output.exists()
