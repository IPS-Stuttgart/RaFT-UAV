"""Compatibility package fixing MMUAD image-evidence target accounting.

The maintained implementation lives in the sibling ``image_evidence.py`` module.
This package preserves the public import surface while ensuring sequence summaries
count the actual target timeline selected for each sequence, including official or
image-derived timestamps used when no truth file is supplied.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "image_evidence.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._image_evidence_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load image-evidence implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


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
    """Build image evidence with target counts from the selected target timeline."""

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
        for target_time_s, image_row in _IMPL._sample_nearest_image_rows(
            image_rows,
            finite_target_times,
            max_frames=max_frames_per_sequence,
            max_time_delta_s=max_image_time_delta_s,
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


def _finite_target_times(values: Any) -> list[float]:
    """Return finite target timestamps while ignoring malformed optional values."""

    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(number):
            result.append(number)
    return result


_IMPL.build_image_evidence = build_image_evidence

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["build_image_evidence"] = build_image_evidence
globals()["_finite_target_times"] = _finite_target_times

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
