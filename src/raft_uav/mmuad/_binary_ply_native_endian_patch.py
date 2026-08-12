"""Normalize binary PLY scalar fields to native byte order before pandas use."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_binary_ply_native_endian_patch_applied"


def _as_native_endian(values: np.ndarray) -> np.ndarray:
    """Return ``values`` in host byte order without copying native arrays."""

    values = np.asarray(values)
    if values.dtype.isnative:
        return values
    return values.astype(values.dtype.newbyteorder("="), copy=False)


def install() -> None:
    """Patch the PLY binary reader so pandas never receives non-native buffers."""

    io_module = import_module("raft_uav.mmuad.io")
    if getattr(io_module, _PATCH_MARKER, False):
        return

    implementation = io_module._impl
    original_reader = implementation._read_binary_ply_payload

    @wraps(original_reader)
    def _read_binary_ply_payload(
        payload: bytes,
        *,
        properties: list[tuple[str, str]],
        vertex_count: int,
        endian: str,
        path: Path,
    ) -> pd.DataFrame:
        dtype_fields = [
            (
                name,
                implementation._ply_numpy_dtype(
                    type_name=type_name,
                    endian=endian,
                ),
            )
            for name, type_name in properties
        ]
        dtype = np.dtype(dtype_fields)
        expected_bytes = int(vertex_count) * dtype.itemsize
        if len(payload) < expected_bytes:
            raise ValueError(f"binary PLY file has incomplete vertex payload: {path}")
        arr = np.frombuffer(
            payload[:expected_bytes],
            dtype=dtype,
            count=int(vertex_count),
        )
        frame = pd.DataFrame(
            {
                name: _as_native_endian(arr[name])
                for name, _type_name in properties
            }
        )
        return implementation._normalize_point_frame(frame, path=path)

    implementation._read_binary_ply_payload = _read_binary_ply_payload
    io_module._read_binary_ply_payload = _read_binary_ply_payload
    setattr(io_module, _PATCH_MARKER, True)
