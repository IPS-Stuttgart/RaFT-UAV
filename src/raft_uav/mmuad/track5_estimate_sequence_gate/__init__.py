"""Compatibility wrapper rejecting ambiguous sequence-gate weight tables.

The maintained implementation lives in the sibling
``track5_estimate_sequence_gate.py`` module. This package preserves the public
import path while rejecting duplicate normalized sequence identifiers instead
of silently applying the last CSV row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_sequence_gate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_sequence_gate_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load estimate sequence-gate implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _sequence_weight_map(rows: pd.DataFrame) -> dict[str, float]:
    """Normalize sequence weights while rejecting duplicate identifiers."""

    frame = pd.DataFrame(rows).copy()
    sequence_column = _IMPL._first_present(frame, _IMPL.SEQUENCE_ALIASES)
    weight_column = _IMPL._first_present(frame, _IMPL.WEIGHT_ALIASES)
    if sequence_column is None or weight_column is None:
        raise ValueError("sequence weight table must contain sequence_id and weight columns")

    result: dict[str, float] = {}
    for _, row in frame.iterrows():
        sequence_id = _IMPL._official_sequence_id_or_none(row[sequence_column])
        if sequence_id is None:
            continue
        if sequence_id in result:
            raise ValueError(
                "sequence weight table contains duplicate normalized sequence_id: "
                f"{sequence_id}"
            )
        result[sequence_id] = _IMPL._validate_weight(
            row[weight_column],
            name="sequence_weight",
        )
    return result


_IMPL._sequence_weight_map = _sequence_weight_map

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence_weight_map"] = _sequence_weight_map

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
