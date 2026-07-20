"""Compatibility fixes for Track 5 submission ensembling.

The maintained implementation lives in the sibling
``track5_submission_ensemble.py`` module. This package preserves the public
import path while preventing malformed normalized classification values from
being silently dropped or truncated to integers and keeping weighted ensemble
arithmetic finite for very large non-negative weights.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.class_probability_context import OFFICIAL_CLASS_LABELS

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_submission_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_submission_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 submission-ensemble implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_INTERNAL_SUBMISSION_ROWS = (
    _IMPL._normalize_internal_submission_rows
)
_ORIGINAL_ENSEMBLE_TRACK5_SUBMISSIONS = _IMPL.ensemble_track5_submissions


def _normalize_internal_submission_rows(
    rows: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    """Reject invalid classes on normalized rows with usable measurements."""

    frame = pd.DataFrame(rows).copy()
    lookup = _IMPL._normalized_column_lookup(frame)
    classification_column = _IMPL._normalized_classification_column(lookup)
    measurement_columns = ("time_s", "state_x_m", "state_y_m", "state_z_m")
    if classification_column is not None and all(
        column in lookup for column in measurement_columns
    ):
        measurements = pd.DataFrame(
            {
                column: pd.to_numeric(frame[lookup[column]], errors="coerce")
                for column in measurement_columns
            },
            index=frame.index,
        )
        finite_measurements = pd.Series(
            np.isfinite(measurements.to_numpy(dtype=float)).all(axis=1),
            index=frame.index,
        )
        if bool(finite_measurements.any()):
            raw_classes = frame.loc[finite_measurements, classification_column]
            normalized_classes = _IMPL._predicted_class_labels(raw_classes)
            valid_classes = normalized_classes.isin(OFFICIAL_CLASS_LABELS)
            if not bool(valid_classes.all()):
                examples = sorted(
                    {repr(value) for value in raw_classes.loc[~valid_classes].tolist()}
                )
                raise ValueError(
                    "invalid normalized Track 5 Classification values in "
                    f"{source_path}: {', '.join(examples)}"
                )

    return _ORIGINAL_NORMALIZE_INTERNAL_SUBMISSION_ROWS(
        frame,
        source_path=source_path,
    )


def ensemble_track5_submissions(
    submissions: Iterable[object],
    *,
    class_policy: str = "weighted-vote",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensemble submissions without overflowing finite non-negative weights.

    Only relative weights affect the ensemble. Converting them to probabilities
    before the legacy implementation prevents both the weight sum and weighted
    position numerator from overflowing. Zero-weight grid entries remain valid,
    while the complete weight vector must retain positive mass. Diagnostic sums
    and vote margins are converted back to the original weight scale afterwards.
    """

    inputs = tuple(submissions)
    if not inputs:
        return _ORIGINAL_ENSEMBLE_TRACK5_SUBMISSIONS(
            inputs,
            class_policy=class_policy,
        )

    weights = np.asarray([float(item.weight) for item in inputs], dtype=float)
    if not np.isfinite(weights).all() or bool(np.any(weights < 0.0)):
        raise ValueError("submission weights must be non-negative and finite")

    scale = float(np.max(weights))
    if scale <= 0.0:
        raise ValueError("submission weights must have positive finite mass")
    scaled_weights = weights / scale
    scaled_total = float(np.sum(scaled_weights))
    if not np.isfinite(scaled_total) or scaled_total <= 0.0:
        raise ValueError("submission weights must have positive finite mass")
    normalized_weights = scaled_weights / scaled_total

    normalized_inputs = tuple(
        _IMPL.SubmissionInput(
            label=item.label,
            path=item.path,
            weight=float(weight),
        )
        for item, weight in zip(inputs, normalized_weights, strict=True)
    )
    estimates, diagnostics = _ORIGINAL_ENSEMBLE_TRACK5_SUBMISSIONS(
        normalized_inputs,
        class_policy=class_policy,
    )

    raw_total = scale * scaled_total
    if "ensemble_weight_sum" in estimates.columns:
        estimates["ensemble_weight_sum"] = raw_total
    if "weight_sum" in diagnostics.columns:
        diagnostics["weight_sum"] = raw_total
    if "classification_vote_margin" in diagnostics.columns:
        margins = diagnostics["classification_vote_margin"].to_numpy(dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            margins = (margins * scale) * scaled_total
        diagnostics["classification_vote_margin"] = margins

    return estimates, diagnostics


_IMPL._normalize_internal_submission_rows = _normalize_internal_submission_rows
_IMPL.ensemble_track5_submissions = ensemble_track5_submissions

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_internal_submission_rows"] = (
    _normalize_internal_submission_rows
)
globals()["ensemble_track5_submissions"] = ensemble_track5_submissions

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
