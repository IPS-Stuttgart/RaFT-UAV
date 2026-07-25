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


def _bool_column(rows: pd.DataFrame, column: str) -> pd.Series:
    """Normalize Boolean diagnostics, including serialized numeric flags."""

    if column not in rows.columns:
        return pd.Series(False, index=rows.index, dtype=bool)
    values = pd.Series(rows[column], index=rows.index)
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_mask = numeric.notna()
    normalized = pd.Series(False, index=rows.index, dtype=bool)
    normalized.loc[numeric_mask] = numeric.loc[numeric_mask].eq(1.0)

    text = values.fillna("").astype(str).str.strip().str.casefold()
    normalized.loc[~numeric_mask] = text.loc[~numeric_mask].isin(
        {"1", "true", "t", "yes", "y"}
    )
    return normalized


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
    output_dir.mkdir(parents=True, exist_ok=True)
    snapped, diagnostics = snap_official_results_to_template(
        results,
        template,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        classification_policy=classification_policy,
        missing_position_policy=missing_position_policy,
    )
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

    valid = _bool_column(diagnostics, "valid")
    extrapolated = _bool_column(diagnostics, "extrapolated")
    large_gap_fallback = _bool_column(diagnostics, "large_gap_fallback")
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
