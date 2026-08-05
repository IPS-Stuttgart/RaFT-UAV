"""Compatibility fixes for candidate-assignment diagnostic parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int as _safe_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_assignment_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_assignment_diagnostics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load candidate-assignment diagnostics implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ASSIGNMENT_DIAGNOSTICS = (
    _IMPL.build_candidate_assignment_diagnostics
)


def _assignment_weights(group: pd.DataFrame) -> np.ndarray:
    """Return finite normalized assignment weights.

    Malformed, negative, NaN, and infinite weights carry no usable probability
    mass. If no positive finite mass remains, fall back to a uniform distribution.
    """

    if "mixture_final_weight" in group.columns:
        weights = pd.to_numeric(
            group["mixture_final_weight"], errors="coerce"
        ).to_numpy(dtype=float)
    elif "mixture_dominant" in group.columns:
        weights = np.asarray(
            [
                _IMPL._parse_mixture_dominant_flag(value)
                for value in group["mixture_dominant"]
            ],
            dtype=float,
        )
    else:
        weights = np.ones(len(group), dtype=float)

    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    scale = float(np.max(weights, initial=0.0))
    if scale <= 0.0:
        return np.ones(len(group), dtype=float) / max(float(len(group)), 1.0)

    scaled = weights / scale
    scaled_total = float(np.sum(scaled))
    if not np.isfinite(scaled_total) or scaled_total <= 0.0:
        return np.ones(len(group), dtype=float) / max(float(len(group)), 1.0)
    if scale <= 1.0e-12 / scaled_total:
        return np.ones(len(group), dtype=float) / max(float(len(group)), 1.0)
    return scaled / scaled_total


def _final_truth_snapshots(truth: pd.DataFrame) -> pd.DataFrame:
    """Return the final finite truth row for each normalized sequence timestamp."""

    rows = pd.DataFrame(truth).copy()
    order_column = "__raft_uav_truth_input_order__"
    while order_column in rows.columns:
        order_column = f"_{order_column}"
    rows[order_column] = np.arange(len(rows), dtype=np.int64)
    normalized = _IMPL.normalize_truth_columns(rows)
    if normalized.empty:
        return normalized.drop(columns=[order_column], errors="ignore")
    return (
        normalized.sort_values(
            ["sequence_id", "time_s", order_column],
            kind="mergesort",
        )
        .drop_duplicates(subset=["sequence_id", "time_s"], keep="last")
        .drop(columns=[order_column])
        .reset_index(drop=True)
    )


def build_candidate_assignment_diagnostics(
    assignments: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    config: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build diagnostics using final valid truth snapshots at duplicate times."""

    return _ORIGINAL_BUILD_CANDIDATE_ASSIGNMENT_DIAGNOSTICS(
        assignments,
        _final_truth_snapshots(truth),
        config=config,
    )


# Candidate ranks are integer identifiers. The legacy float round-trip silently
# truncated fractional values and lost precision for integers above 2**53.
_IMPL._safe_int = _safe_int
_IMPL._assignment_weights = _assignment_weights
_IMPL._final_truth_snapshots = _final_truth_snapshots
_IMPL.build_candidate_assignment_diagnostics = build_candidate_assignment_diagnostics

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_safe_int"] = _safe_int
globals()["_assignment_weights"] = _assignment_weights
globals()["_final_truth_snapshots"] = _final_truth_snapshots
globals()["build_candidate_assignment_diagnostics"] = (
    build_candidate_assignment_diagnostics
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
