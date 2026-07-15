"""Compatibility package for support-safe Track 5 ensemble weight search.

The maintained implementation lives in the sibling
``track5_estimate_ensemble_weight_search.py`` module. This package preserves the
public import path while ensuring weight vectors are compared on complete truth
support instead of being rewarded for missing difficult rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / (
    "track5_estimate_ensemble_weight_search.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_ensemble_weight_search_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 weight-search implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SEARCH = _IMPL.search_track5_estimate_ensemble_weights
_ORIGINAL_SELECTION_OBJECTIVE_VALUE = _IMPL._selection_objective_value


def _with_support_metrics(
    metrics: dict[str, Any],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Add explicit truth-support diagnostics to one metric record."""

    result = dict(metrics)
    expected = max(int(expected_rows), 0)
    matched = max(int(result.get("matched_rows", 0)), 0)
    unmatched = max(expected - matched, 0)
    result["truth_rows"] = expected
    result["unmatched_rows"] = unmatched
    result["coverage_fraction"] = (
        float(min(matched, expected)) / float(expected) if expected else np.nan
    )
    return result


def _score_template_estimates(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, Any]:
    """Score estimates and retain the full finite truth support as the denominator."""

    expected_rows = int(len(truth))
    rows = _IMPL._merge_template_estimates_to_truth(estimates, truth)
    if rows.empty:
        return _with_support_metrics(
            _IMPL._empty_metrics(),
            expected_rows=expected_rows,
        )
    estimated_xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(estimated_xyz).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    metrics = (
        _IMPL._metrics_from_errors(
            np.linalg.norm(estimated_xyz[finite] - truth_xyz[finite], axis=1)
        )
        if finite.any()
        else _IMPL._empty_metrics()
    )
    return _with_support_metrics(metrics, expected_rows=expected_rows)


def _score_template_estimates_by_sequence(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Return one score row for every truth sequence, including missing estimates."""

    truth_rows = pd.DataFrame(truth).copy()
    if truth_rows.empty:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                *_IMPL._empty_metrics(),
                "truth_rows",
                "unmatched_rows",
                "coverage_fraction",
            ]
        )
    estimate_rows = pd.DataFrame(estimates).copy()
    if "sequence_id" in estimate_rows.columns:
        estimate_sequence = estimate_rows["sequence_id"].astype(str)
    else:
        estimate_sequence = pd.Series("", index=estimate_rows.index, dtype="object")

    records: list[dict[str, Any]] = []
    for sequence_id, sequence_truth in truth_rows.groupby("sequence_id", sort=True):
        sequence_text = str(sequence_id)
        sequence_estimates = estimate_rows.loc[estimate_sequence == sequence_text]
        records.append(
            {
                "sequence_id": sequence_text,
                **_score_template_estimates(sequence_estimates, sequence_truth),
            }
        )
    return pd.DataFrame.from_records(records)


def _selection_objective_value(
    pooled_metrics: dict[str, Any],
    sequence_metrics: dict[str, Any],
    *,
    selection_objective: str,
    sequence_objective_weight: float,
) -> float:
    """Reject incomplete-support weight vectors before comparing their errors."""

    if int(pooled_metrics.get("unmatched_rows", 0)) > 0:
        return float("inf")
    return _ORIGINAL_SELECTION_OBJECTIVE_VALUE(
        pooled_metrics,
        sequence_metrics,
        selection_objective=selection_objective,
        sequence_objective_weight=sequence_objective_weight,
    )


def search_track5_estimate_ensemble_weights(
    *args: Any,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Search weights and require at least one full-support grid point."""

    grid, best = _ORIGINAL_SEARCH(*args, **kwargs)
    unmatched = pd.to_numeric(grid.get("unmatched_rows"), errors="coerce")
    finite = unmatched[np.isfinite(unmatched.to_numpy(dtype=float, na_value=np.nan))]
    if finite.empty or not bool((finite == 0).any()):
        minimum = int(finite.min()) if not finite.empty else "unknown"
        raise ValueError(
            "weight search found no candidate with complete truth support; "
            f"minimum unmatched rows: {minimum}"
        )
    return grid, best


_IMPL._score_template_estimates = _score_template_estimates
_IMPL._score_template_estimates_by_sequence = _score_template_estimates_by_sequence
_IMPL._selection_objective_value = _selection_objective_value
_IMPL.search_track5_estimate_ensemble_weights = search_track5_estimate_ensemble_weights

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_with_support_metrics"] = _with_support_metrics
globals()["_score_template_estimates"] = _score_template_estimates
globals()["_score_template_estimates_by_sequence"] = _score_template_estimates_by_sequence
globals()["_selection_objective_value"] = _selection_objective_value
globals()["search_track5_estimate_ensemble_weights"] = (
    search_track5_estimate_ensemble_weights
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
