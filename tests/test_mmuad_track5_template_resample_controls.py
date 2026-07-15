from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import (
    resample_estimates_to_track5_template,
    write_track5_template_resample_outputs,
)


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.5],
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.asarray([1.0]),
    ],
)
@pytest.mark.parametrize(
    "field",
    ["max_nearest_time_delta_s", "max_interpolation_gap_s"],
)
def test_resample_rejects_invalid_time_controls(field: str, value: object) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field} must be a finite non-negative number",
    ):
        resample_estimates_to_track5_template(
            _estimates(),
            _template(),
            **{field: value},
        )


@pytest.mark.parametrize(
    "field",
    ["max_nearest_time_delta_s", "max_interpolation_gap_s"],
)
def test_resample_accepts_zero_dimensional_numpy_controls(field: str) -> None:
    resampled, diagnostics = resample_estimates_to_track5_template(
        _estimates(),
        _template(),
        **{field: np.asarray(1.0)},
    )

    assert len(resampled) == 1
    assert diagnostics["valid"].tolist() == [True]


@pytest.mark.parametrize(
    "field",
    ["max_nearest_time_delta_s", "max_interpolation_gap_s"],
)
def test_writer_rejects_invalid_time_controls_before_creating_output(
    tmp_path: Path,
    field: str,
) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError,
        match=rf"{field} must be a finite non-negative number",
    ):
        write_track5_template_resample_outputs(
            estimates=_estimates(),
            template=_template(),
            output_dir=output_dir,
            **{field: -1.0},
        )

    assert not output_dir.exists()
