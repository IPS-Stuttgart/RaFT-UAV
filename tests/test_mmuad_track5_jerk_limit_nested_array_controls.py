from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.track5_jerk_limit import _finite_scalar, _normalize_iterations


def _object_scalar(payload: object) -> np.ndarray:
    value = np.empty((), dtype=object)
    value[()] = payload
    return value


def test_jerk_iterations_reject_nested_non_scalar_array() -> None:
    nested_vector = _object_scalar(np.array([2.0]))

    with pytest.raises(
        ValueError,
        match="iterations must be a positive finite integer",
    ):
        _normalize_iterations(nested_vector)


def test_jerk_scalar_validation_preserves_nested_zero_dimensional_scalars() -> None:
    nested_scalar = _object_scalar(_object_scalar(np.array(2.0)))

    assert _normalize_iterations(nested_scalar) == 2


def test_jerk_scalar_validation_rejects_cyclic_object_arrays() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="invalid control"):
        _finite_scalar(cyclic, message="invalid control")
