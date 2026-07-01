"""Core public API for snapping official MMUAD Track 5 rows to a template."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import OFFICIAL_UG2_RESULT_COLUMNS
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
    result_by_sequence = {
        seq: group.sort_values("Timestamp").reset_index(drop=True)
        for seq, group in _normalize_results_rows(results).groupby("Sequence", sort=True)
    }

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
            classification = 0
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
                max_interpolation_gap_s=max_interpolation_gap_s,
            )
            classification = _resampled_classification(
                source, timestamp, classification_policy=class_policy
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
