"""Compatibility wrapper for soft class-conditioned Track 5 ensembling.

The maintained implementation lives in the sibling
``track5_soft_class_ensemble.py`` module. This package preserves the public
import path while canonicalizing integer-like classifier labels such as ``0.0``
before the legacy implementation constructs one-hot class probabilities.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_soft_class_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_soft_class_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load soft class ensemble implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _predicted_class_labels(values: pd.Series) -> pd.Series:
    """Return canonical official class-id strings from classifier labels."""

    raw = pd.Series(values)
    text = raw.where(raw.notna(), "").astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float)
    boolean_values = raw.map(lambda value: isinstance(value, bool | np.bool_)).to_numpy(bool)
    integer_like = (
        np.isfinite(numeric_array)
        & np.isclose(numeric_array, np.rint(numeric_array))
        & ~boolean_values
    )
    if integer_like.any():
        positions = np.flatnonzero(integer_like)
        text.iloc[positions] = np.rint(numeric_array[positions]).astype(int).astype(str)
    return text


def _validate_predicted_class_labels(labels: pd.Series) -> None:
    """Reject non-empty classifier labels outside the official class IDs."""

    allowed_labels = tuple(_IMPL._official_class_labels())
    text = pd.Series(labels).fillna("").astype(str).str.strip()
    present = text.ne("")
    invalid = present & ~text.isin(allowed_labels)
    if not invalid.any():
        return
    examples = sorted(text.loc[invalid].unique())
    allowed = ", ".join(allowed_labels)
    raise ValueError(
        "predicted_class values must be official Track 5 class IDs "
        f"{{{allowed}}}; got {examples}"
    )


def _normalize_probability_rows(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Normalize soft-class probabilities without losing integer-like labels."""

    rows = pd.DataFrame(probabilities).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id"])
    sequence_column = _IMPL._first_present(rows, _IMPL.SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError("class probabilities must contain sequence_id/Sequence")
    out = pd.DataFrame({"sequence_id": rows[sequence_column].astype(str)})
    found_probability = False
    for label in _IMPL._official_class_labels():
        column = _IMPL._probability_column(rows, label)
        if column is not None:
            out[f"class_prob_{label}"] = pd.to_numeric(rows[column], errors="coerce")
            found_probability = True
    if not found_probability:
        predicted_column = _IMPL._first_present(rows, _IMPL.PREDICTED_CLASS_ALIASES)
        if predicted_column is None:
            raise ValueError("class probabilities need probability columns or predicted_class")
        predicted = _predicted_class_labels(rows[predicted_column])
        _validate_predicted_class_labels(predicted)
        for label in _IMPL._official_class_labels():
            out[f"class_prob_{label}"] = (predicted == label).astype(float)
    out = out.groupby("sequence_id", as_index=False).mean(numeric_only=True)
    return _IMPL._normalize_probability_mass(out)


_IMPL._normalize_probability_rows = _normalize_probability_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep the patched helpers available for focused tests and exploratory use.
globals()["_predicted_class_labels"] = _predicted_class_labels
globals()["_validate_predicted_class_labels"] = _validate_predicted_class_labels
globals()["_normalize_probability_rows"] = _normalize_probability_rows
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
