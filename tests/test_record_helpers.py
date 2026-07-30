from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.record_helpers import record_arrays


def _record() -> dict[str, object]:
    return {
        "time_s": 1.0,
        "state": np.arange(6, dtype=float),
        "covariance": np.eye(6),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time_s", np.complex128(1.0 + 2.0j)),
        ("state", np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0 + 1.0j])),
        ("covariance", np.eye(6, dtype=complex) * (1.0 + 1.0j)),
    ],
)
def test_record_arrays_rejects_complex_values(field: str, value: object) -> None:
    record = _record()
    record[field] = value

    with pytest.raises(ValueError, match=rf"{field} must contain only real values"):
        record_arrays([record])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time_s", np.ma.masked),
        ("state", np.ma.array(np.arange(6, dtype=float), mask=[0, 0, 0, 0, 0, 1])),
        ("covariance", np.ma.array(np.eye(6), mask=np.eye(6, dtype=bool))),
    ],
)
def test_record_arrays_rejects_masked_values(field: str, value: object) -> None:
    record = _record()
    record[field] = value

    message = rf"{field} must (?:not contain masked values|be an unmasked real scalar)"
    with pytest.raises(ValueError, match=message):
        record_arrays([record])


def test_record_arrays_preserves_valid_records() -> None:
    record = _record()

    times, states, covariances = record_arrays([record])

    np.testing.assert_array_equal(times, [1.0])
    np.testing.assert_array_equal(states, [record["state"]])
    np.testing.assert_array_equal(covariances, [record["covariance"]])
