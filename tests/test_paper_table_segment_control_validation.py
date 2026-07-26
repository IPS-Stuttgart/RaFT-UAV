from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.paper_table import run_paper_table_diagnostic


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        np.bool_(False),
        0,
        -1,
        1.5,
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.array([2]),
        pd.NA,
        np.ma.masked,
    ],
)
def test_paper_table_rejects_invalid_stable_segment_min_frames(
    bad_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="stable_segment_min_frames must be a positive integer scalar",
    ):
        run_paper_table_diagnostic(
            dataset_root=Path("missing-dataset"),
            flight_name="missing-flight",
            stable_segment_min_frames=bad_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        np.bool_(False),
        0.0,
        -1.0,
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.array([65.0]),
        pd.NA,
        np.ma.masked,
    ],
)
def test_paper_table_rejects_invalid_stable_segment_transition_speed(
    bad_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "stable_segment_max_transition_speed_mps must be a finite positive "
            "real scalar"
        ),
    ):
        run_paper_table_diagnostic(
            dataset_root=Path("missing-dataset"),
            flight_name="missing-flight",
            stable_segment_max_transition_speed_mps=bad_value,  # type: ignore[arg-type]
        )
