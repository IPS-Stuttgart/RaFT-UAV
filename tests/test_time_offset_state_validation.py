from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset_state import apply_time_offset


@pytest.mark.parametrize(
    "offset_s",
    [
        True,
        False,
        np.bool_(True),
        float("nan"),
        float("inf"),
        -float("inf"),
        "not-a-number",
        None,
    ],
)
def test_apply_time_offset_rejects_malformed_offsets(offset_s: object) -> None:
    frame = pd.DataFrame({"time_s": [1.0, 2.0]})

    with pytest.raises(ValueError, match="offset_s must be a finite numeric value"):
        apply_time_offset(frame, offset_s=offset_s)  # type: ignore[arg-type]

    assert frame["time_s"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("offset_s", "expected"),
    [
        (2, [3.0, 4.0]),
        (np.float64(-0.25), [0.75, 1.75]),
        ("0.5", [1.5, 2.5]),
    ],
)
def test_apply_time_offset_preserves_finite_numeric_compatibility(
    offset_s: object,
    expected: list[float],
) -> None:
    frame = pd.DataFrame({"time_s": [1.0, 2.0]})

    shifted = apply_time_offset(frame, offset_s=offset_s)  # type: ignore[arg-type]

    assert shifted["time_s"].tolist() == expected
    assert frame["time_s"].tolist() == [1.0, 2.0]
