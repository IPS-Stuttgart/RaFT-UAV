"""Compatibility fix preserving opaque Track 5 scorecard sequence identifiers.

The maintained implementation lives in the sibling ``track5_scorecard.py``
module. This package preserves the public import path while loading optional
paper-diagnostic CSV files without numeric inference changing sequence IDs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_scorecard.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_scorecard_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 scorecard implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_SEQUENCE_IDENTIFIER_DTYPES = {
    "sequence_id": "string",
    "sequence": "string",
    "Sequence": "string",
}


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    """Load optional scorecard diagnostics while preserving opaque IDs."""

    if path is None:
        return None
    return pd.read_csv(path, dtype=_SEQUENCE_IDENTIFIER_DTYPES)


_IMPL._load_optional_csv = _load_optional_csv

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_load_optional_csv"] = _load_optional_csv

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
