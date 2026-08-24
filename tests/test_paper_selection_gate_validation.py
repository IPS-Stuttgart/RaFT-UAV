from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.paper_selection import (
    range_gated_radar_candidates,
    require_fortem_range_m,
)


def _radar(*, include_catprob: bool = True) -> pd.DataFrame:
    data: dict[str, list[float | int]] = {
        "time_s": [0.0],
        "track_id": [1],
        "east_m": [100.0],
        "north_m": [0.0],
        "up_m": [0.0],
        "range_m": [100.0],
    }
    if include_catprob:
        data["cat_prob_uav"] = [0.8]
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "range_gate_m",
    [-1.0, np.nan, np.inf, -np.inf, True, [800.0]],
)
def test_paper_range_gate_rejects_malformed_thresholds(range_gate_m: object) -> None:
    with pytest.raises(ValueError, match="range_gate_m must be a finite non-negative"):
        range_gated_radar_candidates(
            _radar(),
            range_gate_m=range_gate_m,  # type: ignore[arg-type]
            require_range_m=True,
        )


@pytest.mark.parametrize(
    "catprob_threshold",
    [-0.01, 1.01, np.nan, np.inf, -np.inf, True, [0.5]],
)
def test_paper_catprob_gate_rejects_malformed_thresholds(
    catprob_threshold: object,
) -> None:
    with pytest.raises(ValueError, match=r"catprob_threshold must be .*\[0, 1\]"):
        range_gated_radar_candidates(
            _radar(include_catprob=False),
            range_gate_m=None,
            catprob_threshold=catprob_threshold,  # type: ignore[arg-type]
            require_range_m=False,
        )


@pytest.mark.parametrize(
    "minimum_finite_fraction",
    [-0.01, 1.01, np.nan, np.inf, -np.inf, True, [0.99]],
)
def test_range_fraction_rejects_malformed_thresholds(
    minimum_finite_fraction: object,
) -> None:
    with pytest.raises(ValueError, match=r"minimum_finite_fraction must be .*\[0, 1\]"):
        require_fortem_range_m(
            _radar(),
            minimum_finite_fraction=minimum_finite_fraction,  # type: ignore[arg-type]
        )


def test_paper_gate_boundaries_remain_valid() -> None:
    radar = _radar()

    selected = range_gated_radar_candidates(
        radar,
        range_gate_m=100.0,
        catprob_threshold=0.8,
        require_range_m=True,
    )
    require_fortem_range_m(radar, minimum_finite_fraction=1.0)

    assert len(selected) == 1
    assert selected["association_range_gate_m"].iloc[0] == 100.0
    assert selected["association_catprob_threshold"].iloc[0] == 0.8
