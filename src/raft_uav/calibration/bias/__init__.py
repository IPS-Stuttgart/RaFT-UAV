"""Compatibility wrapper that scopes bias training alignment by sequence.

The maintained implementation lives in the sibling ``bias.py`` module. This
package preserves the public import surface while preventing pooled flights with
overlapping relative timestamps from borrowing another sequence's truth rows.
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
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load bias implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES = _IMPL.make_bias_training_examples


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed, nullable sequence identifiers for alignment."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    missing = keys.isna() | keys.eq("") | keys.str.lower().isin({"nan", "none", "<na>"})
    return keys.mask(missing)


def _temporary_position_column(columns: pd.Index) -> str:
    name = "__raft_uav_bias_input_position__"
    while name in columns:
        name += "_"
    return name


def make_bias_training_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float = 2.0,
) -> pd.DataFrame:
    """Build bias examples using nearest truth from the same sequence when possible."""

    if "sequence_id" not in measurements.columns or "sequence_id" not in truth.columns:
        return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
            measurements,
            truth,
            source=source,
            target_columns=target_columns,
            time_gate_s=time_gate_s,
        )

    prepared = pd.DataFrame(measurements).copy()
    truth_rows = pd.DataFrame(truth).copy()
    position_column = _temporary_position_column(prepared.columns)
    prepared[position_column] = np.arange(len(prepared), dtype=int)
    measurement_keys = _sequence_keys(prepared["sequence_id"])
    truth_keys = _sequence_keys(truth_rows["sequence_id"])

    parts: list[pd.DataFrame] = []
    for sequence_key in pd.unique(measurement_keys.dropna()):
        measurement_mask = measurement_keys.eq(sequence_key).fillna(False)
        truth_mask = truth_keys.eq(sequence_key).fillna(False)
        if not bool(truth_mask.any()):
            continue
        examples = _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
            prepared.loc[measurement_mask],
            truth_rows.loc[truth_mask],
            source=source,
            target_columns=target_columns,
            time_gate_s=time_gate_s,
        )
        if not examples.empty:
            parts.append(examples)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True, sort=False)
    result = result.sort_values(position_column, kind="mergesort")
    return result.drop(columns=[position_column]).reset_index(drop=True)


_IMPL.make_bias_training_examples = make_bias_training_examples

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence_keys"] = _sequence_keys
globals()["_temporary_position_column"] = _temporary_position_column
globals()["make_bias_training_examples"] = make_bias_training_examples

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
