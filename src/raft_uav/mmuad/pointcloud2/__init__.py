"""Strict PointCloud2 metadata compatibility boundary.

The maintained decoder lives in the sibling ``pointcloud2.py`` module. This
package preserves the public import path while preventing serialized or malformed
metadata from selecting the wrong byte order or invalid dimensions from being
treated as a flat point buffer.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "pointcloud2.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._pointcloud2_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load MMUAD PointCloud2 implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_POINTCLOUD2_TO_DATAFRAME = _IMPL.pointcloud2_to_dataframe
_TRUE_BOOLEAN_TEXT = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_BOOLEAN_TEXT = frozenset({"0", "false", "f", "no", "n", "off"})


class _MessageWithNormalizedEndianness:
    """Delegate message attributes while exposing one normalized Boolean flag."""

    __slots__ = ("_message", "is_bigendian")

    def __init__(self, message: Any, *, is_bigendian: bool) -> None:
        self._message = message
        self.is_bigendian = is_bigendian

    def __getattr__(self, name: str) -> Any:
        return getattr(self._message, name)


def _boolean_metadata(value: Any, *, name: str) -> bool:
    """Return strict Boolean-like scalar metadata without lossy truthiness."""

    error = f"PointCloud2 {name} must be a Boolean scalar"
    seen_array_ids: set[int] = set()
    while isinstance(value, np.ndarray):
        if np.ma.is_masked(value) or value.ndim != 0:
            raise ValueError(error)
        array_id = id(value)
        if array_id in seen_array_ids:
            raise ValueError(error)
        seen_array_ids.add(array_id)
        value = value.item()

    if np.ma.is_masked(value):
        raise ValueError(error)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOLEAN_TEXT:
            return True
        if normalized in _FALSE_BOOLEAN_TEXT:
            return False
        raise ValueError(error)
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(error)

    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    item = scalar.item()
    if np.ma.is_masked(item) or isinstance(item, (bool, np.bool_)):
        raise ValueError(error)
    try:
        numeric = float(item)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if np.isfinite(numeric) and numeric in (0.0, 1.0):
        return bool(int(numeric))
    raise ValueError(error)


def _validate_nonnegative_dimensions(message: Any) -> None:
    """Reject invalid dimensions without disabling zero-dimension compatibility."""

    for name, default in (("width", 0), ("height", 1)):
        value = _IMPL._integer_metadata(getattr(message, name, default), name=name)
        if value < 0:
            raise ValueError(f"PointCloud2 {name} must be non-negative")


def pointcloud2_to_dataframe(message: Any):
    """Decode a PointCloud2-like message after strict metadata validation."""

    _validate_nonnegative_dimensions(message)
    is_bigendian = _boolean_metadata(
        getattr(message, "is_bigendian", False),
        name="is_bigendian",
    )
    normalized_message = _MessageWithNormalizedEndianness(
        message,
        is_bigendian=is_bigendian,
    )
    return _ORIGINAL_POINTCLOUD2_TO_DATAFRAME(normalized_message)


_IMPL.pointcloud2_to_dataframe = pointcloud2_to_dataframe

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_boolean_metadata"] = _boolean_metadata
globals()["_validate_nonnegative_dimensions"] = _validate_nonnegative_dimensions
globals()["pointcloud2_to_dataframe"] = pointcloud2_to_dataframe

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
