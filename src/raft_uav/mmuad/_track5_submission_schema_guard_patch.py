"""Reject ambiguous Track 5 files instead of changing schemas after validation fails."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import pandas as pd

_PATCH_MARKER = "_raft_uav_rejects_ambiguous_track5_dual_schema"
_OFFICIAL_COLUMNS = frozenset(
    {
        "sequence",
        "timestamp",
        "position",
        "classification",
    }
)


def _contains_complete_official_schema(rows: object) -> bool:
    """Return whether all official Track 5 columns are physically present."""

    columns = {
        str(column).strip().casefold()
        for column in pd.DataFrame(rows).columns
    }
    return _OFFICIAL_COLUMNS <= columns


def install() -> None:
    """Keep official-schema validation authoritative for mixed-schema files."""

    from raft_uav.mmuad import track5_submission_ensemble as ensemble

    original: Callable[[Any], bool] = ensemble._has_normalized_submission_columns
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _has_normalized_submission_columns(rows: object) -> bool:
        if _contains_complete_official_schema(rows):
            return False
        return bool(original(rows))

    setattr(_has_normalized_submission_columns, _PATCH_MARKER, True)
    ensemble._has_normalized_submission_columns = _has_normalized_submission_columns

    implementation = getattr(ensemble, "_IMPL", None)
    if implementation is not None:
        implementation._has_normalized_submission_columns = (
            _has_normalized_submission_columns
        )
