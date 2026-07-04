"""Compatibility wrapper for MMUAD submission helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from raft_uav.mmuad import _submission_impl as _impl

_parse_original = _impl.parse_official_classification_cell
_row_diagnostics_original = _impl._official_track5_row_diagnostics

_DIAGNOSTIC_COLUMNS = (
    "row_type",
    "row_index",
    "sequence_id",
    "timestamp",
    "status",
    "reason",
    "classification",
    "x",
    "y",
    "z",
)
_NORMALIZED_COLUMNS = (
    "row_index",
    "sequence_id",
    "timestamp",
    "x",
    "y",
    "z",
    "classification",
)


def _parse_official_classification_cell_with_domain(value: Any) -> int:
    class_id = _parse_original(value)
    if class_id not in _impl.OFFICIAL_TRACK5_CLASS_IDS:
        raise ValueError(f"invalid official Track 5 class id: {class_id!r}")
    return class_id


def _ensure_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    ordered = [*columns, *[column for column in frame.columns if column not in columns]]
    return frame.reindex(columns=ordered)


def _official_track5_row_diagnostics_with_empty_columns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics, normalized = _row_diagnostics_original(frame)
    return (
        _ensure_columns(diagnostics, _DIAGNOSTIC_COLUMNS),
        _ensure_columns(normalized, _NORMALIZED_COLUMNS),
    )


def _validate_official_track5_frame_with_initialized_duplicates(
    frame: pd.DataFrame,
    *,
    template: pd.DataFrame | None,
    timestamp_tolerance_s: float,
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = list(frame.columns)
    if columns != list(_impl.OFFICIAL_UG2_RESULT_COLUMNS):
        errors.append(
            "mmaud_results.csv columns must exactly equal "
            f"{list(_impl.OFFICIAL_UG2_RESULT_COLUMNS)}"
        )
    diagnostics, normalized = _impl._official_track5_row_diagnostics(frame)
    invalid_sequence_count = int((diagnostics["status"] == "invalid_sequence").sum())
    invalid_timestamp_count = int((diagnostics["status"] == "invalid_timestamp").sum())
    invalid_position_count = int((diagnostics["status"] == "invalid_position").sum())
    invalid_classification_count = int(
        (diagnostics["status"] == "invalid_classification").sum()
    )
    duplicate_indices: set[int] = set()
    duplicate_count = 0
    extra_count = 0
    missing_count = 0
    template_count = None
    template_rows: pd.DataFrame | None = None
    if not normalized.empty:
        duplicate_indices = _impl._duplicate_prediction_indices(
            normalized,
            timestamp_tolerance_s=timestamp_tolerance_s,
        )
        duplicate_count = len(duplicate_indices)
        if duplicate_indices:
            diagnostics.loc[
                diagnostics["row_index"].isin(duplicate_indices)
                & diagnostics["status"].eq("ok"),
                "status",
            ] = "duplicate_prediction"
    if template is not None:
        try:
            template_rows = _impl._normalize_track5_template(template)
        except ValueError as exc:
            errors.append(str(exc))
            template_rows = pd.DataFrame(columns=["sequence_id", "time_s"])
        template_count = int(len(template_rows))
        if not template_rows.empty:
            coverage = _impl._track5_template_coverage_rows(
                normalized,
                template_rows,
                timestamp_tolerance_s=timestamp_tolerance_s,
                ignored_prediction_indices=duplicate_indices,
            )
            missing_count = int((coverage["status"] == "missing_template_timestamp").sum())
            extra_indices = set(
                coverage.loc[coverage["status"] == "extra_prediction", "row_index"]
                .dropna()
                .astype(int)
                .tolist()
            )
            extra_count = len(extra_indices)
            if extra_indices:
                diagnostics.loc[
                    diagnostics["row_index"].isin(extra_indices)
                    & diagnostics["status"].eq("ok"),
                    "status",
                ] = "extra_prediction"
            diagnostics = pd.concat([diagnostics, coverage], ignore_index=True, sort=False)
    valid_row_count = int((diagnostics["status"] == "ok").sum())
    summary = {
        "columns": columns,
        "row_count": int(len(frame)),
        "valid_row_count": valid_row_count,
        "invalid_sequence_count": invalid_sequence_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "invalid_position_count": invalid_position_count,
        "invalid_classification_count": invalid_classification_count,
        "duplicate_prediction_count": int(duplicate_count),
        "template_timestamp_count": template_count,
        "missing_template_timestamp_count": int(missing_count),
        "extra_prediction_count": int(extra_count),
        "sequences": _impl._official_track5_validation_sequence_summaries(
            diagnostics,
            template_checked=template is not None,
            template_rows=template_rows,
        ),
    }
    return summary, diagnostics


_impl.parse_official_classification_cell = _parse_official_classification_cell_with_domain
_impl._official_track5_row_diagnostics = _official_track5_row_diagnostics_with_empty_columns
_impl._validate_official_track5_frame = _validate_official_track5_frame_with_initialized_duplicates

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
parse_official_classification_cell = _parse_official_classification_cell_with_domain
