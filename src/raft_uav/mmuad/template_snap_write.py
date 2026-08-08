"""File-output helpers for MMUAD Track 5 template snapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import validate_official_track5_submission
from raft_uav.mmuad.template_snap_core import (
    _normalize_max_interpolation_gap_s,
    snap_official_results_to_template,
)
from raft_uav.mmuad.template_snap_utils import (
    DIAGNOSTICS_CSV,
    MANIFEST_JSON,
    OFFICIAL_ZIP,
    RESULTS_CSV,
    VALIDATION_JSON,
    VALIDATION_ROWS_CSV,
    ClassificationPolicy,
    MissingPositionPolicy,
    ResampleMethod,
    _normalize_template_rows,
)


_INVALID_BOOLEAN = object()
_TRUE_BOOLEAN_TOKENS = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_BOOLEAN_TOKENS = frozenset({"false", "f", "no", "n", "off"})
_MISSING_BOOLEAN_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def _boolean_number(value: float) -> bool | object:
    """Parse one finite numeric Boolean representation."""

    if np.isnan(value):
        return False
    if not np.isfinite(value):
        return _INVALID_BOOLEAN
    if value == 0.0:
        return False
    if value == 1.0:
        return True
    return _INVALID_BOOLEAN


def _boolean_cell(value: object) -> bool | object:
    """Parse one persisted diagnostic flag without using generic truthiness."""

    if value is None or value is pd.NA or np.ma.is_masked(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _TRUE_BOOLEAN_TOKENS:
            return True
        if text in _FALSE_BOOLEAN_TOKENS or text in _MISSING_BOOLEAN_TOKENS:
            return False
        try:
            return _boolean_number(float(text))
        except (TypeError, ValueError, OverflowError):
            return _INVALID_BOOLEAN

    try:
        array = np.asanyarray(value)
    except (TypeError, ValueError):
        return _INVALID_BOOLEAN
    if array.ndim != 0 or np.iscomplexobj(array):
        return _INVALID_BOOLEAN
    if np.ma.isMaskedArray(array) and bool(np.ma.getmaskarray(array).any()):
        return False

    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    try:
        if bool(pd.isna(scalar)):
            return False
    except (TypeError, ValueError):
        return _INVALID_BOOLEAN
    try:
        return _boolean_number(float(scalar))
    except (TypeError, ValueError, OverflowError):
        return _INVALID_BOOLEAN


def _bool_column(rows: pd.DataFrame, column: str) -> pd.Series:
    """Return strict Boolean diagnostics while preserving the row index."""

    if column not in rows.columns:
        return pd.Series(False, index=rows.index, name=column, dtype=bool)

    values = pd.Series(rows[column], index=rows.index, copy=False)
    normalized: list[bool] = []
    invalid: list[tuple[object, object]] = []
    for index, value in values.items():
        parsed = _boolean_cell(value)
        if parsed is _INVALID_BOOLEAN:
            invalid.append((index, value))
            normalized.append(False)
        else:
            normalized.append(bool(parsed))

    if invalid:
        preview = ", ".join(
            f"index {index!r}: {value!r}" for index, value in invalid[:5]
        )
        suffix = "" if len(invalid) <= 5 else f"; plus {len(invalid) - 5} more"
        raise ValueError(
            f"{column} contains invalid Boolean values: {preview}{suffix}"
        )
    return pd.Series(
        normalized,
        index=values.index,
        name=values.name,
        dtype=bool,
    )


def write_template_snapped_submission(
    *,
    results: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    resample_method: ResampleMethod = "linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy: ClassificationPolicy = "sequence-mode",
    missing_position_policy: MissingPositionPolicy = "zero",
) -> dict[str, Path]:
    """Write snapped official CSV/ZIP, validation rows, and a manifest."""

    max_interpolation_gap_s = _normalize_max_interpolation_gap_s(
        max_interpolation_gap_s
    )
    snapped, diagnostics = snap_official_results_to_template(
        results,
        template,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        classification_policy=classification_policy,
        missing_position_policy=missing_position_policy,
    )
    valid = _bool_column(diagnostics, "valid")
    extrapolated = _bool_column(diagnostics, "extrapolated")
    large_gap_fallback = _bool_column(diagnostics, "large_gap_fallback")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "official_results_csv": output_dir / RESULTS_CSV,
        "official_zip": output_dir / OFFICIAL_ZIP,
        "diagnostics_csv": output_dir / DIAGNOSTICS_CSV,
        "validation_json": output_dir / VALIDATION_JSON,
        "validation_rows_csv": output_dir / VALIDATION_ROWS_CSV,
        "manifest_json": output_dir / MANIFEST_JSON,
    }
    csv_text = snapped.to_csv(index=False)
    paths["official_results_csv"].write_text(csv_text, encoding="utf-8")
    with ZipFile(paths["official_zip"], "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(RESULTS_CSV, csv_text)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    validation = validate_official_track5_submission(
        paths["official_zip"], template=template, require_zip=True
    )
    paths["validation_json"].write_text(json.dumps(_jsonable(validation.summary), indent=2))
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)

    manifest = {
        "schema": "raft-uav-mmuad-template-snap-v1",
        "row_count": int(len(snapped)),
        "template_row_count": int(len(_normalize_template_rows(template))),
        "source_result_rows": int(len(results)),
        "valid_snapped_rows": int(valid.sum()),
        "invalid_snapped_rows": int((~valid).sum()),
        "extrapolated_rows": int(extrapolated.sum()),
        "large_gap_fallback_rows": int(large_gap_fallback.sum()),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "resample_method": str(resample_method),
        "classification_policy": str(classification_policy),
        "missing_position_policy": str(missing_position_policy),
        "max_interpolation_gap_s": max_interpolation_gap_s,
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2))
    return paths


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
