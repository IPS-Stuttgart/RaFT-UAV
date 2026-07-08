"""Small normalized schemas for MMUAD-style UAV tracking experiments."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


CANONICAL_CANDIDATE_COLUMNS = (
    "sequence_id",
    "time_s",
    "source",
    "track_id",
    "x_m",
    "y_m",
    "z_m",
    "std_xy_m",
    "std_z_m",
    "confidence",
    "class_name",
)

CANONICAL_TRUTH_COLUMNS = (
    "sequence_id",
    "time_s",
    "x_m",
    "y_m",
    "z_m",
)

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sequence_id": ("sequence", "seq", "scene", "scene_id", "clip", "clip_id"),
    "time_s": ("timestamp_s", "t", "time", "sec", "seconds"),
    "source": ("sensor", "modality", "frame_id", "header.frame_id"),
    "track_id": (
        "track",
        "id",
        "object_id",
        "cluster_id",
        "instance_id",
        "detection_id",
        "child_frame_id",
    ),
    "x_m": (
        "x",
        "east_m",
        "pos_x",
        "position_x",
        "center_x",
        "bbox_center_x",
        "cx",
        "px",
        "point.x",
        "position.x",
        "pose.position.x",
        "pose.pose.position.x",
        "translation.x",
        "transform.translation.x",
        "center.position.x",
        "bbox.center.position.x",
        "bbox.center.x",
        "location.x",
        "coordinates.x",
    ),
    "y_m": (
        "y",
        "north_m",
        "pos_y",
        "position_y",
        "center_y",
        "bbox_center_y",
        "cy",
        "py",
        "point.y",
        "position.y",
        "pose.position.y",
        "pose.pose.position.y",
        "translation.y",
        "transform.translation.y",
        "center.position.y",
        "bbox.center.position.y",
        "bbox.center.y",
        "location.y",
        "coordinates.y",
    ),
    "z_m": (
        "z",
        "up_m",
        "pos_z",
        "position_z",
        "center_z",
        "bbox_center_z",
        "cz",
        "pz",
        "point.z",
        "position.z",
        "pose.position.z",
        "pose.pose.position.z",
        "translation.z",
        "transform.translation.z",
        "center.position.z",
        "bbox.center.position.z",
        "bbox.center.z",
        "location.z",
        "coordinates.z",
    ),
    "std_xy_m": ("xy_std_m", "std_m", "position_std_m", "sigma_xy_m"),
    "std_z_m": ("z_std_m", "sigma_z_m"),
    "confidence": (
        "score",
        "probability",
        "cat_prob",
        "catprob",
        "hypothesis.score",
        "result.score",
        "result.hypothesis.score",
        "results.0.score",
        "results.0.hypothesis.score",
        "results[0].score",
        "results[0].hypothesis.score",
    ),
    "class_name": (
        "uav_type",
        "class",
        "label",
        "category",
        "class_id",
        "hypothesis.class_id",
        "hypothesis.id",
        "result.class_id",
        "result.id",
        "result.hypothesis.class_id",
        "result.hypothesis.id",
        "results.0.class_id",
        "results.0.id",
        "results.0.hypothesis.class_id",
        "results.0.hypothesis.id",
        "results[0].class_id",
        "results[0].id",
        "results[0].hypothesis.class_id",
        "results[0].hypothesis.id",
    ),
}

_TIME_SECOND_ALIASES = (
    "time_s",
    "timestamp_s",
    "stamp_s",
    "t",
    "time",
    "timestamp",
    "stamp",
    "sec",
    "secs",
    "seconds",
    "stamp.sec",
    "header.stamp.sec",
)
_TIME_UNIT_ALIASES = {
    "timestamp_ns": 1.0e-9,
    "time_ns": 1.0e-9,
    "stamp_ns": 1.0e-9,
    "nanoseconds": 1.0e-9,
    "timestamp_us": 1.0e-6,
    "time_us": 1.0e-6,
    "stamp_us": 1.0e-6,
    "timestamp_usec": 1.0e-6,
    "time_usec": 1.0e-6,
    "stamp_usec": 1.0e-6,
    "microseconds": 1.0e-6,
    "timestamp_ms": 1.0e-3,
    "time_ms": 1.0e-3,
    "stamp_ms": 1.0e-3,
    "milliseconds": 1.0e-3,
}
_TIME_SECOND_NANOSECOND_PAIRS = (
    ("sec", "nanosec"),
    ("sec", "nsec"),
    ("secs", "nsecs"),
    ("seconds", "nanoseconds"),
    ("stamp_sec", "stamp_nanosec"),
    ("stamp_secs", "stamp_nsecs"),
    ("stamp.sec", "stamp.nanosec"),
    ("stamp.sec", "stamp.nsec"),
    ("stamp.secs", "stamp.nsecs"),
    ("header.stamp.sec", "header.stamp.nanosec"),
    ("header.stamp.sec", "header.stamp.nsec"),
    ("header.stamp.secs", "header.stamp.nsecs"),
    ("timestamp_sec", "timestamp_nanosec"),
    ("timestamp_secs", "timestamp_nsecs"),
    ("timestamp.sec", "timestamp.nanosec"),
    ("timestamp.sec", "timestamp.nsec"),
    ("timestamp.secs", "timestamp.nsecs"),
)


@dataclass(frozen=True)
class CandidateFrame:
    """Normalized candidate detections for one or more MMUAD sequences."""

    rows: pd.DataFrame

    def validate(self) -> None:
        missing = {"sequence_id", "time_s", "source", "x_m", "y_m", "z_m"}.difference(
            self.rows.columns
        )
        if missing:
            raise ValueError(f"candidate rows missing required columns: {sorted(missing)}")


@dataclass(frozen=True)
class TruthFrame:
    """Normalized UAV ground-truth positions for one or more MMUAD sequences."""

    rows: pd.DataFrame

    def validate(self) -> None:
        missing = {"sequence_id", "time_s", "x_m", "y_m", "z_m"}.difference(self.rows.columns)
        if missing:
            raise ValueError(f"truth rows missing required columns: {sorted(missing)}")


def normalize_candidate_columns(
    frame: pd.DataFrame,
    *,
    default_sequence_id: str = "default",
    default_source: str = "candidate",
) -> pd.DataFrame:
    """Return a normalized candidate table with canonical column names."""

    out = normalize_time_column_aliases(frame.copy(), target="time_s")
    out = _rename_aliases(out)
    if out.empty:
        return pd.DataFrame(columns=CANONICAL_CANDIDATE_COLUMNS)
    if "sequence_id" not in out.columns:
        out["sequence_id"] = default_sequence_id
    if "source" not in out.columns:
        out["source"] = default_source
    if "track_id" not in out.columns:
        out["track_id"] = np.nan
    if "std_xy_m" not in out.columns:
        out["std_xy_m"] = 10.0
    if "std_z_m" not in out.columns:
        out["std_z_m"] = out["std_xy_m"]
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    if "class_name" not in out.columns:
        out["class_name"] = "uav"
    missing_required = {"time_s", "x_m", "y_m", "z_m"}.difference(out.columns)
    if missing_required:
        raise ValueError(
            f"candidate table missing required columns: {sorted(missing_required)}; "
            f"available={list(out.columns)}"
        )
    for col in ("time_s", "x_m", "y_m", "z_m", "std_xy_m", "std_z_m", "confidence"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sequence_id"] = _normalize_sequence_id_values(
        out["sequence_id"],
        default_sequence_id=default_sequence_id,
    )
    out["source"] = _normalize_text_values(
        out["source"],
        default_text=default_source,
    )
    if "track_id" in out.columns:
        out["track_id"] = _normalize_optional_id_values(out["track_id"])
    out = out.loc[np.isfinite(out[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()
    return out.sort_values(["sequence_id", "time_s", "source"]).reset_index(drop=True)


def normalize_truth_columns(
    frame: pd.DataFrame,
    *,
    default_sequence_id: str = "default",
) -> pd.DataFrame:
    """Return a normalized truth table with canonical column names."""

    out = normalize_time_column_aliases(frame.copy(), target="time_s")
    out = _rename_aliases(out)
    if out.empty:
        return pd.DataFrame(columns=CANONICAL_TRUTH_COLUMNS)
    if "sequence_id" not in out.columns:
        out["sequence_id"] = default_sequence_id
    for col in ("time_s", "x_m", "y_m", "z_m"):
        if col not in out.columns:
            raise ValueError(f"truth table missing {col!r}; available={list(out.columns)}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sequence_id"] = _normalize_sequence_id_values(
        out["sequence_id"],
        default_sequence_id=default_sequence_id,
    )
    out = out.loc[np.isfinite(out[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()
    return out.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _normalize_sequence_id_values(
    values: pd.Series,
    *,
    default_sequence_id: str,
) -> pd.Series:
    """Return row-wise sequence ids, filling missing or blank entries."""

    return _normalize_text_values(values, default_text=default_sequence_id)


def _normalize_text_values(
    values: pd.Series,
    *,
    default_text: str,
) -> pd.Series:
    """Return stripped text values, filling missing-like entries."""

    default = str(default_text)
    text = values.where(values.notna(), default).astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, default)


def _normalize_optional_id_values(values: pd.Series) -> pd.Series:
    """Return optional ids, preserving values but making blanks null."""

    out = values.copy()
    text = values.where(values.notna(), "").astype(str).str.strip().str.lower()
    missing = values.isna() | text.eq("") | text.isin({"nan", "none", "<na>"})
    out.loc[missing] = np.nan
    return out


def _column_key(column: object) -> str:
    return str(column).strip().lower()


def _column_lookup(columns: Iterable[object]) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for column in columns:
        lookup.setdefault(_column_key(column), column)
    return lookup


def _rename_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    lower_to_original = _column_lookup(frame.columns)
    rename: dict[Any, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue
        original = lower_to_original.get(_column_key(canonical))
        if original is not None:
            rename[original] = canonical
            continue
        for alias in aliases:
            original = lower_to_original.get(_column_key(alias))
            if original is not None:
                rename[original] = canonical
                break
    return frame.rename(columns=rename)


def normalize_time_column_aliases(
    frame: pd.DataFrame,
    *,
    target: str = "time_s",
) -> pd.DataFrame:
    """Populate ``target`` from common exported timestamp-unit columns.

    When the target column already exists, missing/non-numeric target values are
    filled row-wise from aliases instead of returning early. This prevents valid
    detections or truth rows from being dropped merely because an export included
    a sparse canonical ``time_s`` column next to a complete timestamp alias.
    """

    out = frame.copy()
    lower_to_original = _column_lookup(out.columns)
    target_original = lower_to_original.get(_column_key(target))
    existing = (
        _seconds_or_stamp_dict_series(out[target_original])
        if target_original is not None
        else None
    )

    alias_series = _time_alias_series(out, lower_to_original)
    if existing is not None:
        out[target] = existing if alias_series is None else existing.fillna(alias_series)
        if target_original != target and target_original in out.columns:
            out = out.drop(columns=[target_original])
        return out
    if alias_series is not None:
        out[target] = alias_series
    return out


def _time_alias_series(
    frame: pd.DataFrame,
    lower_to_original: dict[str, object],
) -> pd.Series | None:
    candidates: list[pd.Series] = []
    for seconds_alias, nanoseconds_alias in _TIME_SECOND_NANOSECOND_PAIRS:
        seconds_col = lower_to_original.get(_column_key(seconds_alias))
        nanoseconds_col = lower_to_original.get(_column_key(nanoseconds_alias))
        if seconds_col is None or nanoseconds_col is None:
            continue
        seconds = pd.to_numeric(frame[seconds_col], errors="coerce")
        nanoseconds = pd.to_numeric(frame[nanoseconds_col], errors="coerce").fillna(0.0)
        candidates.append(seconds + nanoseconds * 1.0e-9)
    for alias, scale in _TIME_UNIT_ALIASES.items():
        original = lower_to_original.get(_column_key(alias))
        if original is not None:
            candidates.append(pd.to_numeric(frame[original], errors="coerce") * scale)
    for alias in _TIME_SECOND_ALIASES:
        original = lower_to_original.get(_column_key(alias))
        if original is not None and _column_key(original) != "time_s":
            candidates.append(_seconds_or_stamp_dict_series(frame[original]))
    for alias in ("header.stamp", "header"):
        original = lower_to_original.get(_column_key(alias))
        if original is not None:
            candidates.append(_seconds_or_stamp_dict_series(frame[original]))
    return _combine_time_alias_series(candidates)


def _combine_time_alias_series(candidates: Iterable[pd.Series]) -> pd.Series | None:
    combined: pd.Series | None = None
    for candidate in candidates:
        if not candidate.notna().any():
            continue
        combined = candidate if combined is None else combined.fillna(candidate)
    return combined


def _seconds_or_stamp_dict_series(values: pd.Series) -> pd.Series:
    """Return seconds from scalar values, falling back to ROS stamp dictionaries."""

    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.to_numeric(values.map(_stamp_dict_to_seconds), errors="coerce")
    return numeric.fillna(parsed)


def _stamp_dict_to_seconds(value: Any) -> float | None:
    value = _coerce_stamp_mapping(value)
    if value is None:
        return None

    nested = _mapping_get_case_insensitive(value, "stamp")
    if nested is not None:
        nested_time = _stamp_dict_to_seconds(nested)
        if nested_time is not None:
            return nested_time

    seconds = _first_mapping_value_case_insensitive(value, ("sec", "secs", "seconds"))
    nanoseconds = _first_mapping_value_case_insensitive(
        value,
        ("nanosec", "nsec", "nsecs", "nanoseconds"),
    )
    if seconds is not None:
        try:
            nanosecond_value = 0.0 if _is_json_missing_scalar(nanoseconds) else float(nanoseconds)
            return float(seconds) + nanosecond_value * 1.0e-9
        except (TypeError, ValueError):
            return None

    for alias, scale in _TIME_UNIT_ALIASES.items():
        scalar = _mapping_get_case_insensitive(value, alias)
        if scalar is None:
            continue
        try:
            return float(scalar) * scale
        except (TypeError, ValueError):
            return None

    scalar = _first_mapping_value_case_insensitive(
        value,
        ("time_s", "timestamp_s", "timestamp", "stamp", "time"),
    )
    try:
        return None if scalar is None else float(scalar)
    except (TypeError, ValueError):
        return None


def _coerce_stamp_mapping(value: Any) -> dict[Any, Any] | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (SyntaxError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _mapping_get_case_insensitive(mapping: dict[Any, Any], key: str) -> Any | None:
    for candidate, value in mapping.items():
        if str(candidate).lower() == key.lower():
            return value
    return None


def _first_mapping_value_case_insensitive(
    mapping: dict[Any, Any],
    keys: Iterable[str],
) -> Any | None:
    for key in keys:
        value = _mapping_get_case_insensitive(mapping, key)
        if value is not None:
            return value
    return None


def load_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalar containers into JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): load_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [load_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if _is_json_missing_scalar(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return load_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _is_json_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False
