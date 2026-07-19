"""Compatibility package that ranks valid local leaderboard scores first.

The maintained implementation lives in the sibling ``leaderboard.py`` module.
This package preserves the public import surface while preventing evaluator-
blocked partial scores from outranking complete leaderboard-valid submissions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "leaderboard.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._leaderboard_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load leaderboard implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_LEADERBOARD_ENTRIES_FROM_CONFIG = _IMPL.leaderboard_entries_from_config
_LEGACY_BUILD_MMUAD_LEADERBOARD = _IMPL.build_mmuad_leaderboard


def _validate_unique_method_labels(entries: list[Any]) -> None:
    """Reject duplicate method labels before evaluation summaries can be overwritten."""

    seen: set[Any] = set()
    duplicates: list[Any] = []
    for entry in entries:
        method = entry.method
        if method in seen and method not in duplicates:
            duplicates.append(method)
        seen.add(method)
    if duplicates:
        labels = ", ".join(repr(method) for method in duplicates)
        raise ValueError(
            "leaderboard method labels must be unique; "
            f"duplicate labels: {labels}"
        )


def leaderboard_entries_from_config(
    payload: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> list[Any]:
    """Build config entries while rejecting labels that would overwrite summaries."""

    entries = _LEGACY_LEADERBOARD_ENTRIES_FROM_CONFIG(payload, base_dir=base_dir)
    _validate_unique_method_labels(entries)
    return entries


def build_mmuad_leaderboard(
    entries: Any,
    *,
    rank_metric: str = _IMPL.DEFAULT_RANK_METRIC,
) -> Any:
    """Evaluate entries only after confirming every method label is unique."""

    entry_list = list(entries)
    _validate_unique_method_labels(entry_list)
    return _LEGACY_BUILD_MMUAD_LEADERBOARD(
        entry_list,
        rank_metric=rank_metric,
    )


def rank_leaderboard_frame(
    frame: pd.DataFrame,
    *,
    rank_metric: str = _IMPL.DEFAULT_RANK_METRIC,
) -> pd.DataFrame:
    """Rank valid entries before evaluator-blocked diagnostic scores.

    When validity metadata are available, the ranking metric is selected from
    valid rows only. Blocked rows remain in the output for diagnostics, but they
    sort after every valid entry and cannot force a metric that valid rows do not
    provide.
    """

    if frame.empty:
        return frame.assign(rank=[])

    work = frame.copy()
    eligibility = _leaderboard_eligibility_mask(work)
    metric_rows = (
        work.loc[eligibility]
        if eligibility is not None and bool(eligibility.any())
        else work
    )
    metric = (
        rank_metric
        if _IMPL._column_has_finite_values(metric_rows, rank_metric)
        else _IMPL._fallback_rank_metric(metric_rows)
    )

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if eligibility is not None:
        work["_leaderboard_eligible"] = eligibility.to_numpy(dtype=bool)
        sort_columns.append("_leaderboard_eligible")
        ascending.append(False)
    sort_columns.append(metric)
    ascending.append(True)
    for candidate, asc in (
        ("p95_3d_m", True),
        ("max_3d_m", True),
        ("uav_type_accuracy", False),
        ("method", True),
    ):
        if candidate in work.columns and candidate != metric:
            sort_columns.append(candidate)
            ascending.append(asc)

    work = work.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )
    work = work.drop(columns=["_leaderboard_eligible"], errors="ignore")
    work.insert(0, "rank", range(1, len(work) + 1))
    work["rank_metric"] = metric
    return work.reset_index(drop=True)


def _leaderboard_eligibility_mask(frame: pd.DataFrame) -> pd.Series | None:
    """Return normalized leaderboard validity, or ``None`` for legacy rows."""

    validity_columns = [
        column
        for column in ("score_valid_for_leaderboard", "leaderboard_ready")
        if column in frame.columns
    ]
    if not validity_columns:
        return None

    eligibility = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    for column in validity_columns:
        parsed = frame[column].map(_optional_leaderboard_flag_value).astype("boolean")
        eligibility = eligibility.fillna(parsed)
    return eligibility.fillna(False).astype(bool)


def _optional_leaderboard_flag_value(value: Any) -> bool | None:
    """Return ``None`` for missing metadata so legacy flags can fill the row."""

    if value is None or value is pd.NA:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return _leaderboard_flag_value(value)


def _leaderboard_flag_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) == 1
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric == 1.0)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes"}:
            return True
        try:
            numeric = float(text)
        except ValueError:
            return False
        return bool(np.isfinite(numeric) and numeric == 1.0)
    return False


def _nonnegative_finite_config_float(value: Any, field: str) -> float:
    """Return a finite non-negative real scalar without array coercion."""

    message = f"leaderboard config {field} must be a finite non-negative number"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or scalar.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        numeric = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(message)
    return numeric


_IMPL.leaderboard_entries_from_config = leaderboard_entries_from_config
_IMPL.build_mmuad_leaderboard = build_mmuad_leaderboard
_IMPL.rank_leaderboard_frame = rank_leaderboard_frame
_IMPL._nonnegative_finite_config_float = _nonnegative_finite_config_float

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["leaderboard_entries_from_config"] = leaderboard_entries_from_config
globals()["build_mmuad_leaderboard"] = build_mmuad_leaderboard
globals()["rank_leaderboard_frame"] = rank_leaderboard_frame
globals()["_validate_unique_method_labels"] = _validate_unique_method_labels
globals()["_leaderboard_eligibility_mask"] = _leaderboard_eligibility_mask
globals()["_optional_leaderboard_flag_value"] = _optional_leaderboard_flag_value
globals()["_leaderboard_flag_value"] = _leaderboard_flag_value
globals()["_nonnegative_finite_config_float"] = _nonnegative_finite_config_float

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
