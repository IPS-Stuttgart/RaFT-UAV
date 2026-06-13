"""Safety guard for official Track 5 submission validation.

The public validator marks blank sequence ids and non-finite timestamps with
row-level statuses.  This module keeps the exported validation result
consistent by also surfacing those statuses in the summary and validity flag.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd

_submission = importlib.import_module("raft_uav.mmuad.submission")
_ORIGINAL_VALIDATE_OFFICIAL_TRACK5_SUBMISSION = _submission.validate_official_track5_submission


def _status_count(rows: pd.DataFrame, status: str) -> int:
    if rows.empty or "status" not in rows.columns:
        return 0
    return int((rows["status"].astype(str) == status).sum())


def validate_official_track5_submission(
    path: Path,
    *,
    template: pd.DataFrame | None = None,
    timestamp_tolerance_s: float = 1.0e-6,
    require_zip: bool = True,
) -> Any:
    """Validate a Track 5 submission without overlooking invalid ids/timestamps."""

    validation = _ORIGINAL_VALIDATE_OFFICIAL_TRACK5_SUBMISSION(
        path,
        template=template,
        timestamp_tolerance_s=timestamp_tolerance_s,
        require_zip=require_zip,
    )
    summary = dict(validation.summary)
    invalid_sequence_count = _status_count(validation.rows, "invalid_sequence")
    invalid_timestamp_count = _status_count(validation.rows, "invalid_timestamp")
    summary["invalid_sequence_count"] = invalid_sequence_count
    summary["invalid_timestamp_count"] = invalid_timestamp_count
    summary["valid"] = bool(
        summary.get("valid", False)
        and invalid_sequence_count == 0
        and invalid_timestamp_count == 0
    )
    return _submission.OfficialTrack5Validation(summary=summary, rows=validation.rows)


_submission.validate_official_track5_submission = validate_official_track5_submission
