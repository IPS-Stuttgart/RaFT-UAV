"""Core public API for snapping official MMUAD Track 5 rows to a template."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    OFFICIAL_TRACK5_CLASS_IDS,
    OFFICIAL_UG2_RESULT_COLUMNS,
    parse_official_classification_cell,
    parse_official_sequence_cell,
)
from raft_uav.mmuad.template_snap_utils import (
    CLASSIFICATION_POLICIES,
    DIAGNOSTIC_COLUMNS,
    MISSING_POSITION_POLICIES,
    RESAMPLE_METHODS,
    ClassificationPolicy,
    MissingPositionPolicy,
    ResampleMethod,
    _diagnostic_record,
    _format_position,
    _normalize_choice,
    _normalize_results_rows,
    _normalize_template_rows,
    _resampled_classification,
    _resampled_position,
)


def snap_official_results_to_template(
    results: pd.DataFrame,
    template: pd.DataFrame,
    *,
    resample_method: ResampleMethod = "linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy: ClassificationPolicy = "sequence-mode",
    missing_position_policy: MissingPositionPolicy = "zero",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return official rows snapped to a template and per-row diagnostics."""

    method = _normalize_choice(resample_method, RESAMPLE_METHODS, "resample_method")
    class_policy = _normalize_choice(
        classification_policy, CLASSIFICATION_POLICIES, "classification_policy"
    )
    missing_policy = _normalize_choice(
        missing_position_policy, MISSING_POSITION_POLICIES, "missing_position_policy"
    )
    max_gap_s = _normalize_max_interpolation_gap_s(max_interpolation_gap_s)
    normalized_results = _normalize_results_rows(results)
    _validate_official_classification_ids(normalized_results)
    result_by_sequence = {
        seq: group.sort_values("Timestamp").reset_index(drop=True)
        for seq, group in normalized_results.groupby("Sequence", sort=True)
    }
    template_classes = _template_classification_by_key(template)

    outputs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for template_index, row in _normalize_template_rows(template).iterrows():
        sequence_id = str(row["Sequence"])
        timestamp = float(row["Timestamp"])
        source = result_by_sequence.get(sequence_id)
        if source is None or source.empty:
            if missing_policy == "raise":
                raise ValueError(f"no source results for template sequence {sequence_id!r}")
            position = np.zeros(3, dtype=float)
            classification = template_classes.get((sequence_id, timestamp), 0)
            diagnostic = _diagnostic_record(
                template_index=template_index,
                sequence_id=sequence_id,
                timestamp=timestamp,
                source_row_count=0,
                nearest_time_delta_s=np.nan,
                extrapolated=True,
                method="missing-zero",
                interpolation_gap_s=np.nan,
                large_gap_fallback=False,
                classification_policy=class_policy,
                valid=False,
            )
        else:
            position, interp = _resampled_position(
                source,
                timestamp,
                resample_method=method,
                max_interpolation_gap_s=max_gap_s,
            )
            classification = _resampled_classification(
                _template_time_sampling_rows(source),
                timestamp,
                classification_policy=class_policy,
            )
            diagnostic = _diagnostic_record(
                template_index=template_index,
                sequence_id=sequence_id,
                timestamp=timestamp,
                source_row_count=len(source),
                nearest_time_delta_s=interp["nearest_time_delta_s"],
                extrapolated=interp["extrapolated"],
                method=interp["method"],
                interpolation_gap_s=interp["interpolation_gap_s"],
                large_gap_fallback=interp["large_gap_fallback"],
                classification_policy=class_policy,
                valid=bool(np.isfinite(position).all()),
            )
        outputs.append(
            {
                "Sequence": sequence_id,
                "Timestamp": timestamp,
                "Position": _format_position(position),
                "Classification": int(classification),
            }
        )
        diagnostics.append(diagnostic)

    return (
        pd.DataFrame.from_records(outputs, columns=list(OFFICIAL_UG2_RESULT_COLUMNS)),
        pd.DataFrame.from_records(diagnostics, columns=list(DIAGNOSTIC_COLUMNS)),
    )


def _template_time_sampling_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the source-row view used for template-time nearest sampling."""

    return rows.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")


def _template_classification_by_key(template: pd.DataFrame) -> dict[tuple[str, float], int]:
    """Return optional template classifications keyed by normalized sequence/time."""

    frame = pd.DataFrame(template)
    lower = {str(column).strip().lower(): column for column in frame.columns}
    sequence_column = lower.get("sequence") or lower.get("sequence_id")
    timestamp_column = lower.get("timestamp") or lower.get("time_s")
    classification_column = lower.get("classification") or lower.get("class_id")
    if (
        frame.empty
        or sequence_column is None
        or timestamp_column is None
        or classification_column is None
    ):
        return {}

    rows = pd.DataFrame(
        {
            "Sequence": frame[sequence_column].map(_template_sequence_key),
            "Timestamp": pd.to_numeric(frame[timestamp_column], errors="coerce"),
            "Classification": frame[classification_column].map(_template_classification_value),
        }
    )
    finite = (
        rows["Sequence"].notna()
        & np.isfinite(rows["Timestamp"].to_numpy(float))
        & rows["Classification"].notna()
    )
    rows = rows.loc[finite]
    return {
        (str(row["Sequence"]), float(row["Timestamp"])): int(row["Classification"])
        for _, row in rows.iterrows()
    }


def _template_sequence_key(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _template_classification_value(value: Any) -> int | None:
    try:
        class_id = parse_official_classification_cell(value)
    except ValueError:
        return None
    if class_id not in OFFICIAL_TRACK5_CLASS_IDS:
        return None
    return class_id


def _validate_official_classification_ids(rows: pd.DataFrame) -> None:
    invalid = ~rows["Classification"].isin(OFFICIAL_TRACK5_CLASS_IDS)
    if not invalid.any():
        return
    bad_value = rows.loc[invalid, "Classification"].iloc[0]
    allowed = ", ".join(str(item) for item in sorted(OFFICIAL_TRACK5_CLASS_IDS))
    raise ValueError(
        "official MMUAD Classification values must be one of "
        f"{{{allowed}}}; got {bad_value!r}"
    )


def _normalize_max_interpolation_gap_s(value: float | None) -> float | None:
    if value is None:
        return None
    gap_s = float(value)
    if not np.isfinite(gap_s) or gap_s < 0.0:
        raise ValueError("max_interpolation_gap_s must be a finite non-negative number")
    return gap_s
