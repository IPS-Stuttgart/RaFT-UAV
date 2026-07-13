"""Compatibility wrapper for robust Track 5 probability relabeling.

The maintained implementation lives in the sibling
``track5_classification_relabel.py`` module. This package preserves the public
import path while rejecting sequence-level probability groups that contain no
positive finite mass instead of assigning an arbitrary uniform fallback class.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_classification_relabel.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_classification_relabel_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 classification relabel implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SEQUENCE_PREDICTION_LABELS = _IMPL._sequence_prediction_labels


def _sequence_prediction_labels(sequence_predictions: pd.DataFrame) -> pd.DataFrame:
    """Build sequence labels without fabricating classes from empty probability mass."""

    rows = pd.DataFrame(sequence_predictions).copy()
    if rows.empty:
        return _ORIGINAL_SEQUENCE_PREDICTION_LABELS(rows)
    sequence_column = _IMPL._first_present(rows, _IMPL.SEQUENCE_ALIASES)
    if sequence_column is None:
        return _ORIGINAL_SEQUENCE_PREDICTION_LABELS(rows)
    probability_items = _IMPL._probability_columns(rows)
    probability_class_ids = tuple(class_id for class_id, _column in probability_items)
    if not _IMPL._valid_probability_class_ids(probability_class_ids):
        return _ORIGINAL_SEQUENCE_PREDICTION_LABELS(rows)

    rows["Sequence"] = rows[sequence_column].astype(str).str.strip()
    probability_columns = [column for _class_id, column in probability_items]
    probability_rows = rows[["Sequence", *probability_columns]].copy()
    for column in probability_columns:
        probability_rows[column] = pd.to_numeric(
            probability_rows[column],
            errors="coerce",
        )
    grouped = probability_rows.groupby("Sequence", sort=True)[probability_columns].mean()
    values = grouped.to_numpy(dtype=float)
    positive_finite = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    totals = positive_finite.sum(axis=1)
    invalid = grouped.index[totals <= 0.0].astype(str).tolist()
    if invalid:
        preview = ", ".join(repr(sequence_id) for sequence_id in invalid[:10])
        suffix = "" if len(invalid) <= 10 else f", ... ({len(invalid)} total)"
        raise ValueError(
            "sequence prediction probabilities have no positive finite mass for "
            f"sequence(s): {preview}{suffix}"
        )
    return _ORIGINAL_SEQUENCE_PREDICTION_LABELS(rows)


_IMPL._sequence_prediction_labels = _sequence_prediction_labels

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence_prediction_labels"] = _sequence_prediction_labels

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
