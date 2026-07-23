"""Compatibility guard for unambiguous NIS reliability gate columns.

The maintained implementation lives in the sibling ``nis_reliability.py`` module.
This package preserves the public import path while rejecting gate-probability
sets that would overwrite one another after column-suffix formatting.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "nis_reliability.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._nis_reliability_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load NIS reliability implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NIS_RELIABILITY_SUMMARY = _IMPL.nis_reliability_summary


def _validated_gate_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    """Return valid probabilities whose formatted output suffixes are unique."""

    probabilities = tuple(_IMPL._validate_probability(value) for value in values)
    values_by_suffix: dict[str, list[float]] = {}
    for probability in probabilities:
        suffix = _IMPL._probability_suffix(probability)
        values_by_suffix.setdefault(suffix, []).append(probability)

    collisions = {
        suffix: suffix_values
        for suffix, suffix_values in values_by_suffix.items()
        if len(suffix_values) > 1
    }
    if collisions:
        rendered = "; ".join(
            f"{suffix}: {', '.join(repr(value) for value in suffix_values)}"
            for suffix, suffix_values in sorted(collisions.items())
        )
        raise ValueError(
            "gate probabilities produce duplicate output column suffixes: "
            f"{rendered}"
        )
    return probabilities


def nis_reliability_summary(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = _IMPL.DEFAULT_GROUP_COLUMNS,
    gate_probabilities: Sequence[float] = _IMPL.DEFAULT_GATE_PROBABILITIES,
    accepted_only: bool = False,
) -> pd.DataFrame:
    """Return NIS statistics only when every requested gate has unique columns."""

    validated_probabilities = _validated_gate_probabilities(gate_probabilities)
    return _ORIGINAL_NIS_RELIABILITY_SUMMARY(
        frame,
        group_columns=group_columns,
        gate_probabilities=validated_probabilities,
        accepted_only=accepted_only,
    )


_IMPL.nis_reliability_summary = nis_reliability_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_gate_probabilities"] = _validated_gate_probabilities
globals()["nis_reliability_summary"] = nis_reliability_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
