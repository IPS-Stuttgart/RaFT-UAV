"""Reject ambiguous Track 5 schemas and non-unique speed-limit grids."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import pandas as pd

_SCHEMA_PATCH_MARKER = "_raft_uav_rejects_ambiguous_track5_dual_schema"
_SPEED_LIMIT_PATCH_MARKER = "_raft_uav_rejects_duplicate_track5_speed_limit_times"
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

    columns = {str(column).strip().casefold() for column in pd.DataFrame(rows).columns}
    return _OFFICIAL_COLUMNS <= columns


def _install_submission_schema_guard(ensemble: Any) -> None:
    """Keep official-schema validation authoritative for mixed-schema files."""

    original: Callable[[Any], bool] = ensemble._has_normalized_submission_columns
    if getattr(original, _SCHEMA_PATCH_MARKER, False):
        return

    @wraps(original)
    def _has_normalized_submission_columns(rows: object) -> bool:
        if _contains_complete_official_schema(rows):
            return False
        return bool(original(rows))

    setattr(_has_normalized_submission_columns, _SCHEMA_PATCH_MARKER, True)
    ensemble._has_normalized_submission_columns = _has_normalized_submission_columns

    implementation = getattr(ensemble, "_IMPL", None)
    if implementation is not None:
        implementation._has_normalized_submission_columns = (
            _has_normalized_submission_columns
        )


def _install_speed_limit_timestamp_guard(speed_limit: Any) -> None:
    """Reject duplicate timestamps that the projector cannot physically constrain."""

    original: Callable[[pd.DataFrame], pd.DataFrame] = speed_limit._normalized_submission
    if getattr(original, _SPEED_LIMIT_PATCH_MARKER, False):
        return

    @wraps(original)
    def _normalized_submission(submission: pd.DataFrame) -> pd.DataFrame:
        rows = original(submission)
        duplicate_mask = rows.duplicated(
            subset=["sequence_id", "time_s"],
            keep=False,
        )
        if not bool(duplicate_mask.any()):
            return rows

        duplicate_pairs = rows.loc[
            duplicate_mask,
            ["sequence_id", "time_s"],
        ].drop_duplicates()
        preview_pairs = duplicate_pairs.head(5)
        preview = ", ".join(
            f"({sequence_id!r}, {time_s!r})"
            for sequence_id, time_s in preview_pairs.itertuples(
                index=False,
                name=None,
            )
        )
        suffix = ", ..." if len(duplicate_pairs) > len(preview_pairs) else ""
        raise ValueError(
            "submission contains duplicate timestamps within a sequence; "
            "speed limiting requires one position per sequence and time: "
            f"{preview}{suffix}"
        )

    setattr(_normalized_submission, _SPEED_LIMIT_PATCH_MARKER, True)
    speed_limit._normalized_submission = _normalized_submission


def install() -> None:
    """Install Track 5 schema safeguards idempotently."""

    from raft_uav.mmuad import track5_submission_ensemble as ensemble
    from raft_uav.mmuad import track5_speed_limit as speed_limit

    _install_submission_schema_guard(ensemble)
    _install_speed_limit_timestamp_guard(speed_limit)
