from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import (
    resample_estimates_to_track5_template,
    write_track5_template_resample_outputs,
)


@pytest.mark.parametrize(
    "field",
    ["max_nearest_time_delta_s", "max_interpolation_gap_s"],
)
@pytest.mark.parametrize(
    "masked_value",
    [
        np.ma.masked,
        np.ma.array(0.5, mask=True),
    ],
)
def test_template_resample_rejects_masked_time_controls(
    field: str,
    masked_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field} must be a finite non-negative number",
    ):
        resample_estimates_to_track5_template(
            pd.DataFrame(),
            pd.DataFrame(),
            **{field: masked_value},
        )


@pytest.mark.parametrize(
    "field",
    ["max_nearest_time_delta_s", "max_interpolation_gap_s"],
)
def test_template_resample_writer_rejects_masked_time_controls_before_output(
    tmp_path: Path,
    field: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field} must be a finite non-negative number",
    ):
        write_track5_template_resample_outputs(
            estimates=pd.DataFrame(),
            template=pd.DataFrame(),
            output_dir=tmp_path,
            **{field: np.ma.masked},
        )

    assert not tmp_path.exists() or not any(tmp_path.iterdir())
