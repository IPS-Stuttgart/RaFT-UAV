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

    for column in ("score_valid_for_leaderboard", "leaderboard_ready"):
        if column in frame.columns:
            return frame[column].map(_leaderboard_flag_value).astype(bool)
    return None


def _leaderboard_flag_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) == 1
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric == 1.0)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


_IMPL.rank_leaderboard_frame = rank_leaderboard_frame

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["rank_leaderboard_frame"] = rank_leaderboard_frame
globals()["_leaderboard_eligibility_mask"] = _leaderboard_eligibility_mask
globals()["_leaderboard_flag_value"] = _leaderboard_flag_value

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
