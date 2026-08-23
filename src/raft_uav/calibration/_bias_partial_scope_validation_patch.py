"""Reject partially populated physical-scope metadata in bias calibration."""

from __future__ import annotations

import importlib
from typing import Sequence

import numpy as np
import pandas as pd

_bias = importlib.import_module("raft_uav.calibration.bias")
_ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES = _bias.make_bias_training_examples
_SCOPE_COLUMNS = ("sequence_id", "flight_id")
_MISSING_SCOPE_VALUES = frozenset({"", "nan", "none", "<na>", "nat"})


def _scope_value_present(value: object) -> bool:
    """Return whether one scope value is a usable non-missing identifier."""

    if value is None or value is pd.NA or np.ma.is_masked(value):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False
    return str(value).strip().casefold() not in _MISSING_SCOPE_VALUES


def _validate_no_partial_scope(frame: pd.DataFrame, *, label: str) -> None:
    """Fail closed when a physical-scope column is populated on only some rows."""

    if frame.empty:
        return
    for column in _SCOPE_COLUMNS:
        if column not in frame.columns:
            continue
        present = frame[column].map(_scope_value_present).to_numpy(dtype=bool)
        if bool(present.any()) and not bool(present.all()):
            raise ValueError(
                "pooled bias calibration requires complete physical-scope metadata "
                f"on every {label} row"
            )


def make_bias_training_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float = 2.0,
) -> pd.DataFrame:
    """Validate partial scope columns before delegated scoped matching."""

    _validate_no_partial_scope(measurements, label="measurement")
    _validate_no_partial_scope(truth, label="truth")
    return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
        measurements,
        truth,
        source=source,
        target_columns=target_columns,
        time_gate_s=time_gate_s,
    )


_bias.make_bias_training_examples = make_bias_training_examples
_bias._IMPL.make_bias_training_examples = make_bias_training_examples
