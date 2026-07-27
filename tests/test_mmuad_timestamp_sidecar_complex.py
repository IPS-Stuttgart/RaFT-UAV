from __future__ import annotations

import warnings

import numpy as np
import pytest

from raft_uav.mmuad.sequence import (
    _coerce_timestamp_value,
    _timestamp_sidecar_explicit_map,
)


@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
def test_numpy_timestamp_map_rejects_complex_times(
    tmp_path,
    complex_dtype: type[np.complexfloating],
) -> None:
    sidecar = tmp_path / "frame_timestamps.npz"
    np.savez(
        sidecar,
        filename=np.array(["frame-1.pcd"]),
        time_s=np.array([complex_dtype(12.5 + 3.0j)]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        timestamp_map = _timestamp_sidecar_explicit_map(sidecar)

    assert timestamp_map == {}


def test_timestamp_coercion_rejects_object_wrapped_complex_scalar() -> None:
    value = np.array(np.complex64(12.5 + 3.0j), dtype=object)

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        timestamp = _coerce_timestamp_value(value)

    assert np.isnan(timestamp)
