from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset_state import (
    OnlineTimeOffsetEstimator,
    apply_time_offset,
)


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


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("offset_s", True),
        ("offset_s", np.nan),
        ("offset_s", "1.0"),
        ("variance_s2", 0.0),
        ("variance_s2", np.inf),
        ("process_variance_s2", -1.0),
        ("min_speed_mps", -1.0),
        ("min_speed_mps", np.array([1.0])),
    ],
)
def test_estimator_rejects_invalid_mutated_state_atomically(
    field: str,
    invalid_value: object,
) -> None:
    estimator = OnlineTimeOffsetEstimator(
        offset_s=0.25,
        variance_s2=2.0,
        process_variance_s2=0.01,
        min_speed_mps=0.5,
    )
    upstream_before = (
        estimator._estimator.offset,
        estimator._estimator.variance,
        estimator._estimator.process_variance,
        estimator._estimator.min_speed,
    )
    setattr(estimator, field, invalid_value)

    with pytest.raises(ValueError):
        _ = estimator.std_s

    upstream_after = (
        estimator._estimator.offset,
        estimator._estimator.variance,
        estimator._estimator.process_variance,
        estimator._estimator.min_speed,
    )
    assert upstream_after == upstream_before


def test_estimator_resynchronizes_valid_mutated_state() -> None:
    estimator = OnlineTimeOffsetEstimator()
    estimator.offset_s = np.float64(0.75)
    estimator.variance_s2 = np.float64(4.0)
    estimator.process_variance_s2 = np.float64(0.25)
    estimator.min_speed_mps = np.float64(2.0)

    assert estimator.std_s == pytest.approx(2.0)
    assert estimator._estimator.offset == pytest.approx(0.75)
    assert estimator._estimator.process_variance == pytest.approx(0.25)
    assert estimator._estimator.min_speed == pytest.approx(2.0)
