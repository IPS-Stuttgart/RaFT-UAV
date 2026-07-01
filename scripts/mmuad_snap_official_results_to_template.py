#!/usr/bin/env python
"""Compatibility wrapper for the MMUAD Track 5 template snapper."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.submission import OFFICIAL_UG2_RESULT_COLUMNS  # noqa: E402
from raft_uav.mmuad.template_snap_cli import main  # noqa: E402
from raft_uav.mmuad.template_snap_core import (  # noqa: E402
    snap_official_results_to_template,
)
from raft_uav.mmuad.template_snap_utils import (  # noqa: E402
    CLASSIFICATION_POLICIES,
    DIAGNOSTIC_COLUMNS,
    DIAGNOSTICS_CSV,
    MANIFEST_JSON,
    MISSING_POSITION_POLICIES,
    OFFICIAL_ZIP,
    RESULTS_CSV,
    RESAMPLE_METHODS,
    VALIDATION_JSON,
    VALIDATION_ROWS_CSV,
    ClassificationPolicy,
    MissingPositionPolicy,
    ResampleMethod,
    _bracketing_gap_s,
    _diagnostic_record,
    _format_float,
    _format_position,
    _normalize_choice,
    _normalize_results_rows,
    _normalize_template_rows,
    _resampled_classification,
    _resampled_position,
    load_official_track5_results_frame_from_frame,
)
from raft_uav.mmuad.template_snap_write import (  # noqa: E402
    _jsonable,
    write_template_snapped_submission,
)

__all__ = [
    "CLASSIFICATION_POLICIES",
    "ClassificationPolicy",
    "DIAGNOSTIC_COLUMNS",
    "DIAGNOSTICS_CSV",
    "MANIFEST_JSON",
    "MISSING_POSITION_POLICIES",
    "MissingPositionPolicy",
    "OFFICIAL_UG2_RESULT_COLUMNS",
    "OFFICIAL_ZIP",
    "RESULTS_CSV",
    "RESAMPLE_METHODS",
    "ResampleMethod",
    "VALIDATION_JSON",
    "VALIDATION_ROWS_CSV",
    "_bracketing_gap_s",
    "_diagnostic_record",
    "_format_float",
    "_format_position",
    "_jsonable",
    "_normalize_choice",
    "_normalize_results_rows",
    "_normalize_template_rows",
    "_resampled_classification",
    "_resampled_position",
    "load_official_track5_results_frame_from_frame",
    "main",
    "snap_official_results_to_template",
    "write_template_snapped_submission",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
