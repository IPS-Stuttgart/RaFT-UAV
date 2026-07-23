"""Package wrapper that hardens MMUAD submission CSV and timestamp matching.

The legacy implementation lives in the sibling ``submission.py`` file. This
wrapper preserves the public import path while accepting spreadsheet-exported
class-map and official Track 5 template CSV files with whitespace around alias
headers, rejecting ambiguous class-map headers, validating timestamp tolerances,
and using globally consistent one-to-one template matching.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment

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


def _read_text_csv(source: Any, **kwargs: Any) -> Any:
    """Read a text-valued CSV without converting opaque identifiers."""

    try:
        return _IMPL._impl.pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            **kwargs,
        )
    except TypeError:
        return _IMPL._impl.pd.read_csv(
            source,
            dtype=str,
            na_filter=False,
            **kwargs,
        )


def _physical_csv_columns(path: Path) -> list[str]:
    """Read the unmangled physical header before pandas deduplicates names."""

    header = _read_text_csv(path, header=None, nrows=1)
    if header.empty:
        return []
    return [str(value) for value in header.iloc[0].tolist()]


def _normalized_column_key(value: Any) -> str:
    """Return the lookup key used for class-map aliases."""

    return str(value).strip().casefold()


def _validate_unique_normalized_columns(columns: Any, *, context: str) -> None:
    """Reject headers that collapse to one alias lookup key."""

    columns_by_key: dict[str, list[str]] = {}
    for column in columns:
        columns_by_key.setdefault(_normalized_column_key(column), []).append(str(column))
    collisions = [group for group in columns_by_key.values() if len(group) > 1]
    if not collisions:
        return

    rendered = "; ".join(
        ", ".join(repr(column) for column in group)
        for group in collisions
    )
    raise ValueError(
        f"{context} has ambiguous columns after trimming whitespace "
        f"and ignoring case: {rendered}"
    )


def _validated_timestamp_tolerance(value: Any) -> float:
    """Return a finite non-negative timestamp tolerance scalar."""

    np = _IMPL._impl.np
    scalar = value
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("timestamp_tolerance_s must be non-negative and finite")
    if isinstance(scalar, np.ndarray):
        if scalar.ndim != 0:
            raise ValueError("timestamp_tolerance_s must be non-negative and finite")
        scalar = scalar.item()
    elif isinstance(scalar, np.generic):
        scalar = scalar.item()
    if isinstance(scalar, (bool, complex)):
        raise ValueError("timestamp_tolerance_s must be non-negative and finite")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp_tolerance_s must be non-negative and finite") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError("timestamp_tolerance_s must be non-negative and finite")
    return numeric


def _validate_official_track5_submission_with_finite_tolerance(
    path: Path | str,
    *,
    template: Any | None = None,
    timestamp_tolerance_s: float = 1.0e-6,
    require_zip: bool = True,
) -> Any:
    """Validate Track 5 submissions with a well-defined timestamp tolerance."""

    return _LEGACY_VALIDATE_OFFICIAL_TRACK5_SUBMISSION(
        path,
        template=template,
        timestamp_tolerance_s=_validated_timestamp_tolerance(timestamp_tolerance_s),
        require_zip=require_zip,
    )


def _track5_template_coverage_rows(
    predictions: Any,
    template: Any,
    *,
    timestamp_tolerance_s: float,
    ignored_prediction_indices: set[int] | None = None,
) -> Any:
    """Match template timestamps globally before classifying unused predictions."""

    pd = _IMPL._impl.pd
    np = _IMPL._impl.np
    rows: list[dict[str, Any]] = []
    ignored_indices = set(ignored_prediction_indices or set())
    matched_prediction_indices = set(ignored_indices)
    for sequence_id, group in template.groupby("sequence_id", sort=True):
        seq_predictions = predictions.loc[
            predictions["sequence_id"] == str(sequence_id)
        ].copy()
        available = seq_predictions.loc[
            ~seq_predictions["row_index"].astype(int).isin(ignored_indices)
        ]
        assignment = optimal_timestamp_assignment(
            group["time_s"].to_numpy(float),
            available["timestamp"].to_numpy(float),
            tolerance_s=timestamp_tolerance_s,
        )
        for template_position, (_, template_row) in enumerate(group.iterrows()):
            timestamp = float(template_row["time_s"])
            prediction_position = assignment.get(template_position)
            if prediction_position is None:
                rows.append(
                    {
                        "row_type": "template",
                        "row_index": np.nan,
                        "sequence_id": str(sequence_id),
                        "timestamp": timestamp,
                        "status": "missing_template_timestamp",
                        "reason": "no prediction at requested timestamp",
                    }
                )
                continue
            matched_row = available.iloc[prediction_position]
            row_index = int(matched_row["row_index"])
            matched_prediction_indices.add(row_index)
            rows.append(
                {
                    "row_type": "template",
                    "row_index": row_index,
                    "sequence_id": str(sequence_id),
                    "timestamp": timestamp,
                    "status": "covered_template_timestamp",
                    "reason": "",
                }
            )
    for _, prediction in predictions.iterrows():
        row_index = int(prediction["row_index"])
        if row_index in matched_prediction_indices:
            continue
        rows.append(
            {
                "row_type": "prediction",
                "row_index": row_index,
                "sequence_id": str(prediction["sequence_id"]),
                "timestamp": float(prediction["timestamp"]),
                "status": "extra_prediction",
                "reason": "prediction does not match a requested template timestamp",
            }
        )
    return pd.DataFrame.from_records(rows)


def _normalize_track5_template_with_stripped_headers(template: Any) -> Any:
    """Normalize template rows while tolerating whitespace-padded alias headers."""

    return _LEGACY_NORMALIZE_TRACK5_TEMPLATE(_strip_dataframe_column_whitespace(template))


def _load_sequence_class_map_with_stripped_csv_headers(path: Path | str | None) -> dict[str, str]:
    """Load class maps while accepting padded but rejecting ambiguous CSV headers."""

    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return _LEGACY_LOAD_SEQUENCE_CLASS_MAP(path)

    _validate_unique_normalized_columns(
        _physical_csv_columns(path),
        context="class-map CSV",
    )
    frame = _read_text_csv(path)
    frame = _strip_dataframe_column_whitespace(frame)
    _validate_unique_normalized_columns(frame.columns, context="class-map CSV")

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


_IMPL._impl.load_sequence_class_map = _load_sequence_class_map_with_stripped_csv_headers
_IMPL.load_sequence_class_map = _load_sequence_class_map_with_stripped_csv_headers
_IMPL._impl._normalize_track5_template = _normalize_track5_template_with_stripped_headers
_IMPL._normalize_track5_template = _normalize_track5_template_with_stripped_headers
_IMPL._impl._track5_template_coverage_rows = _track5_template_coverage_rows
if hasattr(_IMPL, "_track5_template_coverage_rows"):
    _IMPL._track5_template_coverage_rows = _track5_template_coverage_rows
_IMPL._impl.validate_official_track5_submission = (
    _validate_official_track5_submission_with_finite_tolerance
)
_IMPL.validate_official_track5_submission = (
    _validate_official_track5_submission_with_finite_tolerance
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

load_sequence_class_map = _load_sequence_class_map_with_stripped_csv_headers
_normalize_track5_template = _normalize_track5_template_with_stripped_headers
_track5_template_coverage_rows = _track5_template_coverage_rows
validate_official_track5_submission = (
    _validate_official_track5_submission_with_finite_tolerance
)
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
