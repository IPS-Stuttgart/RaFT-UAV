"""Compatibility fixes for bias-correction calibration and application.

The maintained implementation lives in the sibling ``bias.py`` module. This
package preserves the public import path while ensuring ``correct_frame``
respects ``keep_uncorrected=False`` and serialized truth timestamps are numeric
before nearest-time calibration sorting.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

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


def make_bias_training_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float = 2.0,
) -> pd.DataFrame:
    """Build bias examples after normalizing serialized truth timestamps."""

    normalized_truth = truth.copy()
    if "time_s" in normalized_truth.columns:
        normalized_truth["time_s"] = pd.to_numeric(
            normalized_truth["time_s"],
            errors="coerce",
        )
    return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
        measurements,
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
globals()["make_bias_training_examples"] = make_bias_training_examples

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
