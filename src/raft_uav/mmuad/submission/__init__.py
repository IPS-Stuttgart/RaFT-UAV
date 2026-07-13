"""Package wrapper that hardens MMUAD submission CSV and tolerance handling.

The legacy implementation lives in the sibling ``submission.py`` file. This
wrapper preserves the public import path while accepting spreadsheet-exported
CSV files with padded alias headers and rejecting invalid Track 5 timestamp
tolerances before coverage matching.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "submission.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._submission_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy MMUAD submission helpers from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_TRACK5_TEMPLATE_ATTR = "_raft_uav_original_normalize_track5_template"

_LEGACY_LOAD_SEQUENCE_CLASS_MAP = _IMPL.load_sequence_class_map
_LEGACY_VALIDATE_OFFICIAL_TRACK5_SUBMISSION = _IMPL.validate_official_track5_submission
if not hasattr(_IMPL._impl, _ORIGINAL_NORMALIZE_TRACK5_TEMPLATE_ATTR):
    setattr(
        _IMPL._impl,
        _ORIGINAL_NORMALIZE_TRACK5_TEMPLATE_ATTR,
        _IMPL._impl._normalize_track5_template,
    )
_LEGACY_NORMALIZE_TRACK5_TEMPLATE = getattr(
    _IMPL._impl,
    _ORIGINAL_NORMALIZE_TRACK5_TEMPLATE_ATTR,
)


def _strip_dataframe_column_whitespace(frame: Any) -> Any:
    """Return a shallow copy with surrounding whitespace removed from column names."""

    out = frame.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out


def _normalize_track5_template_with_stripped_headers(template: Any) -> Any:
    """Normalize template rows while tolerating whitespace-padded alias headers."""

    return _LEGACY_NORMALIZE_TRACK5_TEMPLATE(_strip_dataframe_column_whitespace(template))


def _load_sequence_class_map_with_stripped_csv_headers(
    path: Path | str | None,
) -> dict[str, str]:
    """Load class maps while tolerating whitespace-padded CSV alias headers."""

    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return _LEGACY_LOAD_SEQUENCE_CLASS_MAP(path)

    try:
        frame = _IMPL._impl.pd.read_csv(path, dtype=str, keep_default_na=False)
    except TypeError:
        frame = _IMPL._impl.pd.read_csv(path)
    frame = _strip_dataframe_column_whitespace(frame)

    lower = {str(column).casefold(): column for column in frame.columns}
    rename: dict[Any, str] = {}
    for alias in _IMPL._impl._SEQUENCE_ID_ALIASES:
        column = lower.get(str(alias).casefold())
        if column is not None:
            rename[column] = "sequence_id"
            break
    for alias in _IMPL._impl._UAV_TYPE_ALIASES:
        column = lower.get(str(alias).casefold())
        if column is not None:
            rename[column] = "uav_type"
            break
    frame = frame.rename(columns=rename)
    missing = {"sequence_id", "uav_type"}.difference(frame.columns)
    if missing:
        raise ValueError(f"class-map CSV missing columns: {sorted(missing)}")

    class_map: dict[str, str] = {}
    for _, row in frame.iterrows():
        sequence_id = _IMPL._class_map_sequence_key(row["sequence_id"])
        uav_type = _IMPL._class_map_uav_type(row["uav_type"])
        if sequence_id is not None and uav_type is not None:
            class_map[sequence_id] = uav_type
    return class_map


def _normalize_timestamp_tolerance_s(value: Any) -> float:
    """Return a finite, non-negative Track 5 timestamp tolerance."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "timestamp_tolerance_s must be a finite non-negative number, not a boolean"
        )
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(
                "timestamp_tolerance_s must be a finite non-negative scalar number"
            )
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    try:
        tolerance = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "timestamp_tolerance_s must be a finite non-negative number"
        ) from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("timestamp_tolerance_s must be a finite non-negative number")
    return tolerance


def _validate_official_track5_submission_with_finite_tolerance(
    path: Path | str,
    *,
    template: Any | None = None,
    timestamp_tolerance_s: float = 1.0e-6,
    require_zip: bool = True,
) -> Any:
    """Validate a Track 5 upload with a finite scalar timestamp tolerance."""

    tolerance = _normalize_timestamp_tolerance_s(timestamp_tolerance_s)
    return _LEGACY_VALIDATE_OFFICIAL_TRACK5_SUBMISSION(
        path,
        template=template,
        timestamp_tolerance_s=tolerance,
        require_zip=require_zip,
    )


_IMPL._impl.load_sequence_class_map = _load_sequence_class_map_with_stripped_csv_headers
_IMPL.load_sequence_class_map = _load_sequence_class_map_with_stripped_csv_headers
_IMPL._impl._normalize_track5_template = _normalize_track5_template_with_stripped_headers
_IMPL._normalize_track5_template = _normalize_track5_template_with_stripped_headers
_IMPL._impl.validate_official_track5_submission = (
    _validate_official_track5_submission_with_finite_tolerance
)
_IMPL.validate_official_track5_submission = (
    _validate_official_track5_submission_with_finite_tolerance
)
_IMPL._normalize_timestamp_tolerance_s = _normalize_timestamp_tolerance_s

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

load_sequence_class_map = _load_sequence_class_map_with_stripped_csv_headers
_normalize_track5_template = _normalize_track5_template_with_stripped_headers
_normalize_timestamp_tolerance_s = _normalize_timestamp_tolerance_s
validate_official_track5_submission = (
    _validate_official_track5_submission_with_finite_tolerance
)
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
