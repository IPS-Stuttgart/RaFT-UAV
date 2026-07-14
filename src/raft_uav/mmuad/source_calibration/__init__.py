"""Compatibility package with strict source-calibration alpha-grid validation.

The maintained implementation lives in the sibling ``source_calibration.py``
module. This package preserves the public import path while preventing malformed
shrinkage grids from being silently clipped, discarded, or replaced by defaults.
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
    raise ImportError(f"cannot load source calibration implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _normalize_alpha_grid(
    values: tuple[float, ...] | list[float] | None,
) -> tuple[float, ...]:
    """Return a sorted alpha grid without lossy coercion or silent fallback."""

    if values is None:
        return (1.0,)
    if isinstance(values, (str, bytes)):
        raise ValueError(
            "source_translation_alpha_grid must be an iterable of finite numbers in [0, 1]"
        )
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(
            "source_translation_alpha_grid must be an iterable of finite numbers in [0, 1]"
        ) from exc
    if not items:
        raise ValueError("source_translation_alpha_grid must contain at least one value")

    normalized: list[float] = []
    for index, value in enumerate(items):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "source_translation_alpha_grid values must be finite numbers in [0, 1]; "
                f"invalid value at index {index}: {value!r}"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "source_translation_alpha_grid values must be finite numbers in [0, 1]; "
                f"invalid value at index {index}: {value!r}"
            ) from exc
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(
                "source_translation_alpha_grid values must be finite numbers in [0, 1]; "
                f"invalid value at index {index}: {value!r}"
            )
        normalized.append(numeric)
    return tuple(sorted(set(normalized)))


_IMPL._normalize_alpha_grid = _normalize_alpha_grid

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_alpha_grid"] = _normalize_alpha_grid

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
