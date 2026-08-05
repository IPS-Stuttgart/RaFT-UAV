"""Compatibility package for safe source-calibration transforms and lookup.

The maintained implementation lives in the sibling ``source_calibration.py`` module.
This package preserves the public import path while validating every loaded or fitted
source transform before it can contaminate calibrated candidate coordinates, rejecting
lossy Boolean, complex, masked, or cyclic coefficients, rejecting ambiguous
case-insensitive transform keys, and preventing source-specific transforms from leaking
onto unrelated or broader sources.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "source_calibration.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._source_calibration_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load source-calibration implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SOURCE_TRANSFORM_POST_INIT = _IMPL.SourceTransform.__post_init__


def _validate_real_transform_values(
    value: object,
    *,
    name: str,
    seen: set[int] | None = None,
) -> None:
    """Reject values that NumPy would silently coerce while building a transform."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not contain Boolean values")
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f"{name} must not contain complex values")

    if isinstance(value, np.ma.MaskedArray):
        if bool(np.ma.getmaskarray(value).any()):
            raise ValueError(f"{name} must not contain masked values")
        _validate_real_transform_values(value.data, name=name, seen=seen)
        return

    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            raise ValueError(f"{name} must not contain Boolean values")
        if np.issubdtype(value.dtype, np.complexfloating):
            raise ValueError(f"{name} must not contain complex values")
        if value.dtype != object:
            return
        seen = set() if seen is None else seen
        value_id = id(value)
        if value_id in seen:
            raise ValueError(f"{name} must not contain cyclic values")
        seen.add(value_id)
        try:
            for item in value.flat:
                _validate_real_transform_values(item, name=name, seen=seen)
        finally:
            seen.remove(value_id)
        return

    if isinstance(value, (list, tuple)):
        seen = set() if seen is None else seen
        value_id = id(value)
        if value_id in seen:
            raise ValueError(f"{name} must not contain cyclic values")
        seen.add(value_id)
        try:
            for item in value:
                _validate_real_transform_values(item, name=name, seen=seen)
        finally:
            seen.remove(value_id)


def _validated_source_transform_post_init(self: object) -> None:
    """Normalize a source transform after rejecting lossy coefficients."""

    _validate_real_transform_values(self.linear, name="linear transform")
    _validate_real_transform_values(self.translation_m, name="translation_m")
    _ORIGINAL_SOURCE_TRANSFORM_POST_INIT(self)
    if not np.isfinite(self.linear).all():
        raise ValueError("linear transform must contain only finite values")
    if not np.isfinite(self.translation_m).all():
        raise ValueError("translation_m must contain only finite values")


def _source_lookup_key(value: object) -> str:
    """Return the case-insensitive key used for source-transform lookup."""

    return str(value).casefold()


def _require_unambiguous_source_transform_keys(
    transforms: dict[str, object],
) -> None:
    """Reject blank keys and keys that collapse under case-insensitive lookup."""

    keys_by_normalized_value: dict[str, list[str]] = {}
    for key in transforms:
        rendered = str(key)
        if not rendered.strip():
            raise ValueError("source-calibration transforms must use non-blank source keys")
        keys_by_normalized_value.setdefault(_source_lookup_key(rendered), []).append(rendered)

    collisions = [
        sorted(keys)
        for keys in keys_by_normalized_value.values()
        if len(keys) > 1
    ]
    if not collisions:
        return

    rendered_collisions = "; ".join(
        ", ".join(repr(key) for key in keys)
        for keys in sorted(collisions)
    )
    raise ValueError(
        "source-calibration transforms contain ambiguous case-insensitive keys: "
        f"{rendered_collisions}"
    )


def _is_forward_source_prefix(source_key: str, transform_key: str) -> bool:
    """Return whether ``transform_key`` is a token-boundary prefix of ``source_key``."""

    if not transform_key or source_key == transform_key:
        return False
    if not source_key.startswith(transform_key):
        return False
    boundary_index = len(transform_key)
    return (not transform_key[-1].isalnum()) or (
        not source_key[boundary_index].isalnum()
    )


def _match_source_transform(source: str, transforms: dict[str, object]) -> object | None:
    """Return an exact or longest safe forward-prefix transform for one source.

    A calibration key may match a more specific exported source name at a token
    boundary, for example ``sensor_detail`` matching ``sensor_detail_clusters``.
    It must not match an unrelated alphanumeric continuation such as ``radar2``,
    and the reverse direction remains unsafe: a transform fitted specifically for
    ``sensor_detail`` must not be applied to the broader ``sensor`` source.
    """

    _require_unambiguous_source_transform_keys(transforms)
    source_key = _source_lookup_key(source)
    normalized_transforms = [
        (_source_lookup_key(key), transform)
        for key, transform in transforms.items()
    ]
    for transform_key, transform in normalized_transforms:
        if source_key == transform_key:
            return transform
    matches = [
        (len(transform_key), transform)
        for transform_key, transform in normalized_transforms
        if _is_forward_source_prefix(source_key, transform_key)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


_IMPL.SourceTransform.__post_init__ = _validated_source_transform_post_init
_IMPL._match_source_transform = _match_source_transform

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_source_lookup_key"] = _source_lookup_key
globals()["_require_unambiguous_source_transform_keys"] = (
    _require_unambiguous_source_transform_keys
)
globals()["_is_forward_source_prefix"] = _is_forward_source_prefix
globals()["_match_source_transform"] = _match_source_transform

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
