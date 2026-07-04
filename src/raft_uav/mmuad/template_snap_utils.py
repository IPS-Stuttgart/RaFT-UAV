"""Utility functions for MMUAD Track 5 template snapping."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    OFFICIAL_TRACK5_CLASS_IDS,
    parse_official_position_cell,
    parse_official_sequence_cell,
)

RESAMPLE_METHODS = ("linear", "nearest")
CLASSIFICATION_POLICIES = ("sequence-mode", "nearest")
MISSING_POSITION_POLICIES = ("zero", "raise")
ResampleMethod = Literal["linear", "nearest"]
ClassificationPolicy = Literal["sequence-mode", "nearest"]
MissingPositionPolicy = Literal["zero", "raise"]

RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
DIAGNOSTICS_CSV = "mmuad_template_snap_diagnostics.csv"
VALIDATION_JSON = "mmuad_template_snap_validation.json"
VALIDATION_ROWS_CSV = "mmuad_template_snap_validation_rows.csv"
MANIFEST_JSON = "mmuad_template_snap_manifest.json"
DIAGNOSTIC_COLUMNS = (
    "template_row_index",
    "Sequence",
    "Timestamp",
    "source_row_count",
    "nearest_time_delta_s",
    "abs_nearest_time_delta_s",
    "extrapolated",
    "method",
    "interpolation_gap_s",
    "large_gap_fallback",
    "classification_policy",
    "valid",
)


def _normalize_results_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows = load_official_track5_results_frame_from_frame(results)
    positions = np.asarray(
        [parse_official_position_cell(value) for value in rows["Position"]],
        dtype=float,
    ).reshape((-1, 3))
    rows = rows.copy()
    rows["x"] = positions[:, 0]
    rows["y"] = positions[:, 1]
    rows["z"] = positions[:, 2]
    rows["Classification"] = _integer_classification_values(rows["Classification"]).astype(int)
    return rows.sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def load_official_track5_results_frame_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an in-memory official Track 5 result-like frame."""

    lower = {str(column).strip().lower(): column for column in frame.columns}
    missing = [
        column
        for column in ("sequence", "timestamp", "position", "classification")
        if column not in lower
    ]
    if missing:
        raise ValueError(f"official Track 5 results missing columns: {missing}")
    classification = _integer_classification_values(frame[lower["classification"]])
    rows = pd.DataFrame(
        {
            "Sequence": frame[lower["sequence"]].map(_template_sequence_value),
            "Timestamp": pd.to_numeric(frame[lower["timestamp"]], errors="coerce"),
            "Position": frame[lower["position"]],
            "Classification": classification,
        }
    )
    finite = (
        rows["Sequence"].notna()
        & rows["Timestamp"].notna()
        & rows["Classification"].notna()
    )
    rows = rows.loc[finite].copy()
    rows["Timestamp"] = rows["Timestamp"].astype(float)
    rows["Classification"] = rows["Classification"].astype(int)
    return rows.sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def _integer_classification_values(values: pd.Series) -> pd.Series:
    raw = pd.Series(values)
    boolean_mask = raw.map(lambda value: isinstance(value, (bool, np.bool_)))
    if boolean_mask.any():
        row_index = int(np.flatnonzero(boolean_mask.to_numpy())[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids, not booleans; "
            f"got {bad_value!r}"
        )
    numbers = pd.to_numeric(raw, errors="coerce")
    bad_text_mask = numbers.isna() & raw.notna()
    if bad_text_mask.any():
        row_index = int(np.flatnonzero(bad_text_mask.to_numpy())[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids; "
            f"got {bad_value!r}"
        )
    numeric = numbers.to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    integer_like = finite & np.isclose(
        numeric,
        np.rint(numeric),
        rtol=0.0,
        atol=1.0e-12,
    )
    fractional = finite & ~integer_like
    if fractional.any():
        row_index = int(np.flatnonzero(fractional)[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids; "
            f"got {bad_value!r}"
        )
    invalid_domain = np.zeros_like(finite, dtype=bool)
    if integer_like.any():
        integer_values = np.rint(numeric[integer_like]).astype(int)
        invalid_domain[integer_like] = ~np.isin(
            integer_values,
            list(OFFICIAL_TRACK5_CLASS_IDS),
        )
    if invalid_domain.any():
        row_index = int(np.flatnonzero(invalid_domain)[0])
        class_id = int(np.rint(numeric[row_index]))
        allowed = ", ".join(str(item) for item in sorted(OFFICIAL_TRACK5_CLASS_IDS))
        raise ValueError(
            "official MMUAD Classification values must be one of "
            f"{{{allowed}}}; got {class_id!r}"
        )
    return numbers


def _template_sequence_value(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    lower = {str(column).strip().lower(): column for column in template.columns}
    sequence_col = lower.get("sequence") or lower.get("sequence_id")
    timestamp_col = lower.get("timestamp") or lower.get("time_s")
    if sequence_col is None or timestamp_col is None:
        raise ValueError("template must contain Sequence/Timestamp or sequence_id/time_s")
    rows = pd.DataFrame(
        {
            "Sequence": template[sequence_col].map(_template_sequence_value),
            "Timestamp": pd.to_numeric(template[timestamp_col], errors="coerce"),
        }
    )
    rows = rows.loc[rows["Sequence"].notna() & rows["Timestamp"].notna()]
    return rows.sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def _resampled_position(
    sequence_results: pd.DataFrame,
    timestamp: float,
    *,
    resample_method: ResampleMethod,
    max_interpolation_gap_s: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    work = sequence_results.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
    times = work["Timestamp"].to_numpy(float)
    xyz = work[["x", "y", "z"]].to_numpy(float)
    nearest_index = int(np.argmin(np.abs(times - float(timestamp))))
    nearest_delta = float(float(timestamp) - times[nearest_index])
    extrapolated = bool(timestamp < times[0] or timestamp > times[-1])
    gap_s = _bracketing_gap_s(times, float(timestamp))
    fallback = (
        max_interpolation_gap_s is not None
        and np.isfinite(gap_s)
        and gap_s > float(max_interpolation_gap_s)
    )
    if len(times) == 1 or resample_method == "nearest" or fallback:
        return xyz[nearest_index].astype(float), {
            "nearest_time_delta_s": nearest_delta,
            "extrapolated": extrapolated,
            "method": "nearest" if not fallback else "nearest-large-gap-fallback",
            "interpolation_gap_s": gap_s,
            "large_gap_fallback": bool(fallback),
        }
    position = np.asarray(
        [np.interp(float(timestamp), times, xyz[:, axis]) for axis in range(3)]
    )
    return position, {
        "nearest_time_delta_s": nearest_delta,
        "extrapolated": extrapolated,
        "method": "linear",
        "interpolation_gap_s": gap_s,
        "large_gap_fallback": False,
    }


def _resampled_classification(
    sequence_results: pd.DataFrame,
    timestamp: float,
    *,
    classification_policy: ClassificationPolicy,
) -> int:
    if classification_policy == "sequence-mode":
        mode = sequence_results["Classification"].mode(dropna=True)
        if not mode.empty:
            return int(mode.sort_values().iloc[0])
    times = sequence_results["Timestamp"].to_numpy(float)
    nearest_index = int(np.argmin(np.abs(times - timestamp)))
    return int(sequence_results["Classification"].iloc[nearest_index])


def _diagnostic_record(**items: Any) -> dict[str, Any]:
    nearest_delta = items["nearest_time_delta_s"]
    return {
        "template_row_index": int(items["template_index"]),
        "Sequence": str(items["sequence_id"]),
        "Timestamp": float(items["timestamp"]),
        "source_row_count": int(items["source_row_count"]),
        "nearest_time_delta_s": nearest_delta,
        "abs_nearest_time_delta_s": (
            abs(float(nearest_delta)) if np.isfinite(nearest_delta) else np.nan
        ),
        "extrapolated": bool(items["extrapolated"]),
        "method": str(items["method"]),
        "interpolation_gap_s": items["interpolation_gap_s"],
        "large_gap_fallback": bool(items["large_gap_fallback"]),
        "classification_policy": str(items["classification_policy"]),
        "valid": bool(items["valid"]),
    }


def _bracketing_gap_s(times: np.ndarray, timestamp: float) -> float:
    if len(times) < 2 or timestamp <= times[0] or timestamp >= times[-1]:
        return np.nan
    right = int(np.searchsorted(times, timestamp, side="right"))
    if right >= len(times):
        return np.nan
    return float(times[right] - times[max(0, right - 1)])


def _format_position(position: np.ndarray) -> str:
    x, y, z = [_format_float(value) for value in position]
    return f"({x},{y},{z})"


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _normalize_choice(value: str, choices: tuple[str, ...], name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}")
    return normalized
