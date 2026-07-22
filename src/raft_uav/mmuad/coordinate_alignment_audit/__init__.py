"""Compatibility validation for coordinate-alignment audit time gates.

The maintained implementation lives in the sibling
``coordinate_alignment_audit.py`` module. This package preserves the public
import path while rejecting malformed nearest-time gates before they can
silently widen or empty the diagnostic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "coordinate_alignment_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._coordinate_alignment_audit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load coordinate alignment audit from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_COORDINATE_ALIGNMENT_AUDIT = _IMPL.build_coordinate_alignment_audit


def _normalize_max_time_delta_s(value: object | None) -> float | None:
    """Return a finite non-negative scalar time gate or ``None``."""

    if value is None:
        return None
    message = "max_time_delta_s must be a finite non-negative real scalar or None"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0 or array.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        normalized = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(message)
    return normalized


def build_coordinate_alignment_audit(
    sequence_root: Path,
    truth_path: Path,
    *,
    sequence_glob: str = "*",
    voxel_size_m: float = 0.75,
    min_cluster_points: int = 3,
    max_time_delta_s: float | None = 0.5,
    include_translation_diagnostic: bool = True,
    scales: tuple[float, ...] = (0.001, 0.01, 1.0),
) -> pd.DataFrame:
    """Build the audit after validating the nearest-time matching gate."""

    normalized_gate = _normalize_max_time_delta_s(max_time_delta_s)
    return _ORIGINAL_BUILD_COORDINATE_ALIGNMENT_AUDIT(
        sequence_root,
        truth_path,
        sequence_glob=sequence_glob,
        voxel_size_m=voxel_size_m,
        min_cluster_points=min_cluster_points,
        max_time_delta_s=normalized_gate,
        include_translation_diagnostic=include_translation_diagnostic,
        scales=scales,
    )


def _parse_max_time_delta(value: str) -> float | None:
    """Parse CLI time gates while preserving documented unbounded aliases."""

    text = str(value).strip().lower()
    if text in {"none", "inf", "infinite", "unbounded"}:
        return None
    return _normalize_max_time_delta_s(text)


_IMPL.build_coordinate_alignment_audit = build_coordinate_alignment_audit
_IMPL._parse_max_time_delta = _parse_max_time_delta

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_max_time_delta_s"] = _normalize_max_time_delta_s
globals()["build_coordinate_alignment_audit"] = build_coordinate_alignment_audit
globals()["_parse_max_time_delta"] = _parse_max_time_delta

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
