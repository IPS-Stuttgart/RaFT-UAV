"""Compatibility wrapper for supervised estimate sequence-gate fitting.

The maintained implementation lives in the sibling
``track5_estimate_sequence_gate_fit.py`` module. This package preserves the
public import path while preventing sequences without usable truth supervision
from supplying nearest-neighbor blend weights.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_sequence_gate_fit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_sequence_gate_fit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        "cannot load estimate sequence-gate implementation from "
        f"{_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_MAIN = _IMPL.main
_ORIGINAL_LOSO_WEIGHTS = _IMPL._loso_weights
_ORIGINAL_NEAREST_NEIGHBOR_PREDICT = _IMPL._nearest_neighbor_predict


def _supervised_training_features(
    features: pd.DataFrame,
    *,
    minimum_sequences: int,
    context: str,
) -> pd.DataFrame:
    """Return rows with finite, truth-supported oracle blend weights."""

    rows = pd.DataFrame(features).copy()
    required = {"sequence_id", "sequence_gate_weight"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(
            f"{context} requires training features with columns: {sorted(required)}"
        )

    weights = pd.to_numeric(rows["sequence_gate_weight"], errors="coerce")
    supervised = np.isfinite(weights.to_numpy(float))
    if "matched_rows" in rows.columns:
        matched_rows = pd.to_numeric(
            rows["matched_rows"],
            errors="coerce",
        ).to_numpy(float)
        supervised &= np.isfinite(matched_rows) & (matched_rows > 0.0)
    if "pose_mse_m2" in rows.columns:
        pose_mse = pd.to_numeric(
            rows["pose_mse_m2"],
            errors="coerce",
        ).to_numpy(float)
        supervised &= np.isfinite(pose_mse)

    rows = rows.loc[supervised].copy()
    rows["sequence_gate_weight"] = weights.loc[supervised].to_numpy(float)
    sequence_count = rows["sequence_id"].astype(str).nunique()
    if sequence_count < minimum_sequences:
        noun = "sequence" if minimum_sequences == 1 else "sequences"
        raise ValueError(
            f"{context} requires at least {minimum_sequences} {noun} "
            f"with finite oracle supervision; got {sequence_count}"
        )
    return rows.reset_index(drop=True)


def _loso_weights(features: pd.DataFrame) -> pd.DataFrame:
    """Fit LOSO weights only from sequences with usable oracle targets."""

    supervised = _supervised_training_features(
        features,
        minimum_sequences=2,
        context="LOSO sequence-gate fitting",
    )
    return _ORIGINAL_LOSO_WEIGHTS(supervised)


def _nearest_neighbor_predict(
    train: pd.DataFrame,
    apply: pd.DataFrame,
) -> pd.DataFrame:
    """Predict from the nearest sequence that has a finite oracle target."""

    supervised = _supervised_training_features(
        train,
        minimum_sequences=1,
        context="sequence-gate prediction",
    )
    return _ORIGINAL_NEAREST_NEIGHBOR_PREDICT(supervised, apply)


def main(argv: list[str] | None = None) -> int:
    """Run the legacy CLI with this package's active pandas binding."""

    original_impl_pd = _IMPL.pd
    _IMPL.pd = pd
    try:
        return _ORIGINAL_MAIN(argv)
    finally:
        _IMPL.pd = original_impl_pd


_IMPL.main = main
_IMPL._loso_weights = _loso_weights
_IMPL._nearest_neighbor_predict = _nearest_neighbor_predict

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_supervised_training_features"] = _supervised_training_features
globals()["_loso_weights"] = _loso_weights
globals()["_nearest_neighbor_predict"] = _nearest_neighbor_predict

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
