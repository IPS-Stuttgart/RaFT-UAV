"""Compatibility package with strict image-evidence sampling controls.

The maintained implementation lives in the sibling ``image_evidence.py``
module. This package preserves the public import path while rejecting sampling
controls that would otherwise be silently truncated, treated as unlimited, or
allowed to disable timestamp gating through non-finite values.
"""

from __future__ import annotations

import importlib.util
import numbers
from pathlib import Path
import sys
from typing import Any

import numpy as np

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "image_evidence.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._image_evidence_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load image evidence implementation from {_LEGACY_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_IMAGE_EVIDENCE = _IMPL.build_image_evidence
_ORIGINAL_SAMPLE_NEAREST_IMAGE_ROWS = _IMPL._sample_nearest_image_rows


def _scalar_item(value: Any, *, name: str, contract: str) -> Any:
    """Return one scalar item or raise a field-specific validation error."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be {contract}")
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be {contract}") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be {contract}")
    return array.item()


def _normalize_max_frames(value: Any, *, name: str) -> int:
    """Normalize a non-negative integer frame limit without lossy coercion."""

    contract = "a non-negative integer"
    item = _scalar_item(value, name=name, contract=contract)
    if isinstance(item, numbers.Integral):
        integer = int(item)
    else:
        try:
            numeric = float(item)
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be {contract}") from exc
        if not np.isfinite(numeric) or item != integer:
            raise ValueError(f"{name} must be {contract}")
    if integer < 0:
        raise ValueError(f"{name} must be {contract}")
    return integer


def _normalize_max_time_delta(value: Any, *, name: str) -> float | None:
    """Normalize an optional finite non-negative timestamp tolerance."""

    if value is None:
        return None
    contract = "None or a finite non-negative number"
    item = _scalar_item(value, name=name, contract=contract)
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be {contract}") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be {contract}")
    return numeric


def _sample_nearest_image_rows(
    image_rows,
    target_times,
    *,
    max_frames: int,
    max_time_delta_s: float | None,
):
    """Sample image rows after validating direct helper controls."""

    normalized_max_frames = _normalize_max_frames(max_frames, name="max_frames")
    normalized_max_time_delta = _normalize_max_time_delta(
        max_time_delta_s,
        name="max_time_delta_s",
    )
    return _ORIGINAL_SAMPLE_NEAREST_IMAGE_ROWS(
        image_rows,
        target_times,
        max_frames=normalized_max_frames,
        max_time_delta_s=normalized_max_time_delta,
    )


def build_image_evidence(
    sequence_root: Path,
    *,
    truth_file: Path | None = None,
    sequence_glob: str = "*",
    timestamp_source: str = "image",
    max_frames_per_sequence: int = 32,
    max_image_time_delta_s: float | None = 0.5,
    image_feature_backend: str = "handcrafted",
):
    """Build image evidence with exact sampling-control validation."""

    normalized_max_frames = _normalize_max_frames(
        max_frames_per_sequence,
        name="max_frames_per_sequence",
    )
    normalized_max_time_delta = _normalize_max_time_delta(
        max_image_time_delta_s,
        name="max_image_time_delta_s",
    )
    return _ORIGINAL_BUILD_IMAGE_EVIDENCE(
        sequence_root,
        truth_file=truth_file,
        sequence_glob=sequence_glob,
        timestamp_source=timestamp_source,
        max_frames_per_sequence=normalized_max_frames,
        max_image_time_delta_s=normalized_max_time_delta,
        image_feature_backend=image_feature_backend,
    )


_IMPL._normalize_max_frames = _normalize_max_frames
_IMPL._normalize_max_time_delta = _normalize_max_time_delta
_IMPL._sample_nearest_image_rows = _sample_nearest_image_rows
_IMPL.build_image_evidence = build_image_evidence

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_max_frames"] = _normalize_max_frames
globals()["_normalize_max_time_delta"] = _normalize_max_time_delta
globals()["_sample_nearest_image_rows"] = _sample_nearest_image_rows
globals()["build_image_evidence"] = build_image_evidence

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
