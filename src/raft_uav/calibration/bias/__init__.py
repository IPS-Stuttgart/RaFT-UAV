"""Compatibility fixes for bias-correction calibration and application.

The maintained implementation lives in the sibling ``bias.py`` module. This
package preserves the public import path while ensuring ``correct_frame``
respects ``keep_uncorrected=False``, serialized truth timestamps are numeric
before nearest-time calibration sorting, and genuinely complex calibration
values are not silently reduced to their real components.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "bias.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._bias_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load bias correction utilities from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES = _IMPL.make_bias_training_examples


def _correct_frame(
    self: object,
    frame: pd.DataFrame,
    *,
    keep_uncorrected: bool = True,
) -> pd.DataFrame:
    """Apply the model and optionally omit the retained raw target columns."""

    corrected = self.apply(frame)
    if keep_uncorrected:
        return corrected
    raw_columns = [f"raw_{column}" for column in self.target_columns]
    return corrected.drop(columns=raw_columns, errors="ignore")


def _finite_real_numeric_series(values: pd.Series) -> pd.Series:
    """Coerce numeric values without discarding nonzero imaginary components."""

    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy()
    if not np.iscomplexobj(array):
        return numeric
    real = np.real(array)
    imaginary = np.imag(array)
    return pd.Series(
        np.where(
            np.isfinite(real) & np.isfinite(imaginary) & (imaginary == 0.0),
            real,
            np.nan,
        ),
        index=values.index,
        dtype=float,
    )


def _normalized_bias_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy whose requested numeric columns contain finite real values."""

    normalized = frame.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = _finite_real_numeric_series(normalized[column])
    return normalized


def make_bias_training_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float = 2.0,
) -> pd.DataFrame:
    """Build bias examples without silently accepting complex numeric cells."""

    numeric_columns = ("time_s", *(str(column) for column in target_columns))
    normalized_measurements = _normalized_bias_frame(measurements, numeric_columns)
    normalized_truth = _normalized_bias_frame(truth, numeric_columns)
    return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
        normalized_measurements,
        normalized_truth,
        source=source,
        target_columns=target_columns,
        time_gate_s=time_gate_s,
    )


_IMPL.SensorBiasCorrectionModel.correct_frame = _correct_frame
_IMPL.make_bias_training_examples = make_bias_training_examples

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_correct_frame"] = _correct_frame
globals()["_finite_real_numeric_series"] = _finite_real_numeric_series
globals()["_normalized_bias_frame"] = _normalized_bias_frame
globals()["make_bias_training_examples"] = make_bias_training_examples

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
