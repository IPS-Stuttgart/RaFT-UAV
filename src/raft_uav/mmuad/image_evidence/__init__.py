"""Compatibility package with strict image-evidence sampling and target accounting.

The maintained implementation lives in the sibling ``image_evidence.py``
module. This package preserves the public import path while rejecting sampling
controls that would otherwise be silently truncated, treated as unlimited, or
allowed to disable timestamp gating through non-finite values. Sequence
summaries count the actual target timeline selected for each sequence.
"""

from __future__ import annotations

import importlib.util
import numbers
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

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

_ORIGINAL_SAMPLE_NEAREST_IMAGE_ROWS = _IMPL._sample_nearest_image_rows


def _scalar_item(value: Any, *, name: str, contract: str) -> Any:
    """Return one non-Boolean scalar item or raise a field-specific error."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be {contract}")
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be {contract}") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be {contract}")
    item = array.item()
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must be {contract}")
    return item


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


def _finite_target_times(values: Any) -> list[float]:
    """Return the finite target timestamps consumed by the image sampler."""

    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(number):
            result.append(number)
    return result


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
    """Build image evidence with exact controls and actual target counts."""

    normalized_max_frames = _normalize_max_frames(
        max_frames_per_sequence,
        name="max_frames_per_sequence",
    )
    normalized_max_time_delta = _normalize_max_time_delta(
        max_image_time_delta_s,
        name="max_image_time_delta_s",
    )
    backend_requested = _IMPL._normalize_image_feature_backend(image_feature_backend)
    backend_resolved, feature_extractor = _IMPL._make_image_feature_extractor(
        backend_requested
    )
    sequences = _IMPL.discover_sequence_paths(
        Path(sequence_root),
        sequence_glob=sequence_glob,
    )
    truth_by_sequence = _IMPL._truth_times_by_sequence(truth_file)
    frame_records: list[dict[str, Any]] = []
    target_counts: dict[str, int] = {}
    for paths in sequences:
        image_files = _IMPL._sequence_image_files(paths.root)
        if not image_files:
            continue
        image_rows = _IMPL._image_file_rows(image_files)
        if image_rows.empty:
            continue
        target_times = truth_by_sequence.get(paths.sequence_id)
        if target_times is None:
            try:
                target_times = _IMPL.official_track5_sequence_timestamps(
                    paths,
                    timestamp_source=timestamp_source,
                )
            except ValueError:
                target_times = []
        if not target_times:
            target_times = image_rows["image_time_s"].dropna().astype(float).tolist()
        finite_target_times = _finite_target_times(target_times)
        target_counts[str(paths.sequence_id)] = len(finite_target_times)
        for target_time_s, image_row in _sample_nearest_image_rows(
            image_rows,
            finite_target_times,
            max_frames=normalized_max_frames,
            max_time_delta_s=normalized_max_time_delta,
        ):
            record = feature_extractor(Path(image_row["image_path"]))
            record.update(
                {
                    "sequence_id": paths.sequence_id,
                    "target_time_s": float(target_time_s),
                    "image_time_s": float(image_row["image_time_s"]),
                    "image_time_delta_s": float(
                        image_row["image_time_s"] - target_time_s
                    ),
                    "image_path": str(image_row["image_path"]),
                    "image_evidence_mode": _IMPL.IMAGE_EVIDENCE_MODE,
                    "image_feature_backend_requested": backend_requested,
                    "image_feature_backend_resolved": backend_resolved,
                }
            )
            frame_records.append(record)
    frame_features = pd.DataFrame.from_records(frame_records)
    sequence_features = _IMPL._sequence_features_from_frame_features(
        frame_features,
        target_counts=target_counts,
    )
    return _IMPL.ImageEvidenceResult(
        sequence_features=sequence_features,
        frame_features=frame_features,
    )


_IMPL._normalize_max_frames = _normalize_max_frames
_IMPL._normalize_max_time_delta = _normalize_max_time_delta
_IMPL._finite_target_times = _finite_target_times
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
globals()["_finite_target_times"] = _finite_target_times
globals()["_sample_nearest_image_rows"] = _sample_nearest_image_rows
globals()["build_image_evidence"] = build_image_evidence

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
