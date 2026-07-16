from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_geometric_median_ensemble import (
    build_track5_geometric_median_ensemble,
    write_track5_geometric_median_outputs,
)


@pytest.mark.parametrize(
    "weight",
    [
        True,
        np.bool_(False),
        np.array(True),
        np.array([1.0]),
        np.nan,
        np.inf,
        -1.0,
    ],
)
def test_build_rejects_invalid_weight_before_empty_template(weight: object) -> None:
    with pytest.raises(
        ValueError,
        match="estimate weight must be finite and non-negative",
    ):
        build_track5_geometric_median_ensemble(
            [("candidate", pd.DataFrame(), weight)],
            pd.DataFrame(),
        )


def test_build_accepts_zero_dimensional_numeric_weight() -> None:
    estimates, diagnostics = build_track5_geometric_median_ensemble(
        [("candidate", pd.DataFrame(), np.array(1.5))],
        pd.DataFrame(),
    )

    assert estimates.empty
    assert diagnostics.empty


def test_writer_rejects_boolean_weight_before_file_io(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    with pytest.raises(
        ValueError,
        match="estimate weight must be finite and non-negative",
    ):
        write_track5_geometric_median_outputs(
            estimate_inputs=[EstimateInput("missing", tmp_path / "missing.csv", True)],
            template=pd.DataFrame(),
            output_dir=output_dir,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iterations": 0}, "max_iterations must be a positive integer"),
        ({"tolerance_m": np.inf}, "tolerance_m must be a finite non-negative scalar"),
    ],
)
def test_writer_validates_solver_controls_before_file_io(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        write_track5_geometric_median_outputs(
            estimate_inputs=[EstimateInput("missing", tmp_path / "missing.csv", 1.0)],
            template=pd.DataFrame(),
            output_dir=output_dir,
            **kwargs,
        )

    assert not output_dir.exists()
