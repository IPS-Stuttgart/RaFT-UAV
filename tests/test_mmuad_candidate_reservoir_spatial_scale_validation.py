from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_spatial import (
    spatial_diversity_cap_reservoir,
)


@pytest.mark.parametrize(
    "scale",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
    ],
)
def test_spatial_diversity_rejects_invalid_scales(scale: object) -> None:
    with pytest.raises(
        ValueError,
        match="spatial_diversity_scale_m must be a finite positive real scalar",
    ):
        spatial_diversity_cap_reservoir(
            pd.DataFrame(),
            spatial_diversity_scale_m=scale,
        )


def test_spatial_diversity_accepts_zero_dimensional_real_scale() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "source": ["candidate", "candidate"],
            "candidate_branch": ["base", "base"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [1.0, 0.5],
        }
    )

    capped = spatial_diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=2,
        min_per_source=0,
        min_per_branch=0,
        spatial_diversity_scale_m=np.array(5.0),
    )

    assert len(capped) == 2
    assert np.isfinite(capped["candidate_spatial_selection_utility"]).all()
