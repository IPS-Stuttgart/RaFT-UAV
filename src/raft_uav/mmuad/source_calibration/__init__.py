"""Compatibility package for safe source-calibration transforms and lookup.

The maintained implementation lives in the sibling ``source_calibration.py`` module.
This package preserves the public import path while validating every loaded or fitted
source transform before it can contaminate calibrated candidate coordinates and while
preventing source-specific transforms from being applied backward to broader sources.
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


def _validated_source_transform_post_init(self: object) -> None:
    """Normalize a source transform, then reject non-finite coefficients."""

    _ORIGINAL_SOURCE_TRANSFORM_POST_INIT(self)
    if not np.isfinite(self.linear).all():
        raise ValueError("linear transform must contain only finite values")
    if not np.isfinite(self.translation_m).all():
        raise ValueError("translation_m must contain only finite values")


def _match_source_transform(source: str, transforms: dict[str, object]) -> object | None:
    """Return an exact or longest forward-prefix transform for one source.

    A calibration key may match a more specific exported source name, for example
    ``sensor_detail`` matching ``sensor_detail_clusters``.  The reverse is unsafe:
    a transform fitted specifically for ``sensor_detail`` must not be applied to the
    broader ``sensor`` source.
    """

    source_l = str(source).lower()
    for key, transform in transforms.items():
        if source_l == str(key).lower():
            return transform
    matches = [
        (len(str(key)), transform)
        for key, transform in transforms.items()
        if source_l.startswith(str(key).lower())
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

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
