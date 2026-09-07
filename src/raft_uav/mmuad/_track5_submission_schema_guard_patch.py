"""Reject ambiguous Track 5 schemas and unsafe speed-limit output metadata."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import pandas as pd

_SCHEMA_PATCH_MARKER = "_raft_uav_rejects_ambiguous_track5_dual_schema"
_SPEED_LIMIT_OUTPUT_PATCH_MARKER = "_raft_uav_validates_track5_speed_limit_outputs"
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


def _speed_limit_applied_flags(values: pd.Series) -> pd.Series:
    """Parse persisted correction flags without relying on string truthiness."""

    text = values.astype("string").str.strip().str.casefold()
    true = text.isin({"true", "t", "yes", "y", "on"})
    false = text.isna() | text.isin(
        {
            "",
            "false",
            "f",
            "no",
            "n",
            "off",
            "none",
            "null",
            "nan",
            "na",
            "n/a",
            "<na>",
            "nat",
        }
    )
    numeric = pd.to_numeric(text, errors="coerce")
    true = true | numeric.eq(1.0).fillna(False)
    false = false | numeric.eq(0.0).fillna(False)
    invalid = ~(true | false)
    if bool(invalid.any()):
        bad = values.loc[invalid]
        raise ValueError(
            "speed_limit_applied contains invalid Boolean values at rows "
            f"{bad.index[:5].tolist()}: {bad.iloc[:5].tolist()}; expected booleans, "
            "exact numeric 0/1, recognized Boolean text, or missing values"
        )
    return true.astype(bool)


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
        implementation._has_normalized_submission_columns = _has_normalized_submission_columns


def _install_speed_limit_output_guard(speed_limit: Any) -> None:
    """Validate persisted flags and readiness requirements before any output write."""

    original = speed_limit.write_track5_speed_limit_outputs
    if getattr(original, _SPEED_LIMIT_OUTPUT_PATCH_MARKER, False):
        return

    @wraps(original)
    def write_track5_speed_limit_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("require_leaderboard_ready", False) and kwargs.get("template") is None:
            raise ValueError("require_leaderboard_ready=True requires a template")

        diagnostics = kwargs.get("diagnostics")
        if diagnostics is not None:
            normalized = pd.DataFrame(diagnostics).copy()
            if "speed_limit_applied" in normalized.columns:
                flags = _speed_limit_applied_flags(normalized["speed_limit_applied"])
                normalized["speed_limit_applied"] = flags.to_numpy(dtype=bool)
            kwargs["diagnostics"] = normalized

        return original(*args, **kwargs)

    setattr(write_track5_speed_limit_outputs, _SPEED_LIMIT_OUTPUT_PATCH_MARKER, True)
    speed_limit.write_track5_speed_limit_outputs = write_track5_speed_limit_outputs

    implementation = getattr(speed_limit, "_IMPL", None)
    if implementation is not None:
        implementation.write_track5_speed_limit_outputs = write_track5_speed_limit_outputs


def install() -> None:
    """Install Track 5 schema and speed-limit output safeguards idempotently."""

    from raft_uav.mmuad import track5_speed_limit as speed_limit
    from raft_uav.mmuad import track5_submission_ensemble as ensemble

    _install_submission_schema_guard(ensemble)
    _install_speed_limit_output_guard(speed_limit)
