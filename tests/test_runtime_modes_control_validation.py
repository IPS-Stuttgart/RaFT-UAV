from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.runtime_modes import backward_repair_associations


@pytest.mark.parametrize("field", ["max_gap_s", "max_repair_distance_m"])
@pytest.mark.parametrize(
    "value",
    [-1.0, np.nan, np.inf, True, 1 + 0j, np.array([1.0]), np.ma.masked],
)
def test_backward_repair_rejects_invalid_bounds_before_empty_return(
    field: str,
    value: object,
) -> None:
    controls: dict[str, object] = {
        "max_gap_s": 10.0,
        "max_repair_distance_m": 200.0,
    }
    controls[field] = value

    with pytest.raises(ValueError, match=field):
        backward_repair_associations(
            pd.DataFrame(),
            pd.DataFrame(),
            **controls,
        )


def _repair_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.DataFrame(
        {
            "frame_index": [0, 2],
            "time_s": [0.0, 2.0],
            "east_m": [0.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "frame_index": [0, 1, 2],
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )
    return selected, candidates


def test_backward_repair_accepts_scalar_like_bounds_and_exact_matches() -> None:
    selected, candidates = _repair_frames()

    repaired = backward_repair_associations(
        selected,
        candidates,
        max_gap_s=np.array(2.0),
        max_repair_distance_m="0",
    )

    assert repaired["frame_index"].tolist() == [0, 1, 2]
    middle = repaired.loc[repaired["frame_index"] == 1].iloc[0]
    assert bool(middle["association_repaired"])
    assert middle["association_score"] == pytest.approx(0.0)


def test_backward_repair_preserves_zero_gap_bound_semantics() -> None:
    selected, candidates = _repair_frames()

    repaired = backward_repair_associations(
        selected,
        candidates,
        max_gap_s=0.0,
        max_repair_distance_m=10.0,
    )

    assert repaired["frame_index"].tolist() == [0, 2]
