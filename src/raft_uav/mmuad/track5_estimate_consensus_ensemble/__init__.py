"""Compatibility wrapper for padded Track 5 consensus template headers.

The maintained implementation lives in the sibling
``track5_estimate_consensus_ensemble.py`` module. This package preserves the
public import path while making template alias lookup insensitive to surrounding
whitespace, as commonly introduced by spreadsheets and hand-edited CSV files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_consensus_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_consensus_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 consensus ensemble implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _first_present(rows: Any, names: tuple[str, ...]) -> Any | None:
    """Return the original column whose stripped, case-folded name matches."""

    by_normalized_name = {
        str(column).strip().casefold(): column for column in rows.columns
    }
    for name in names:
        if name in rows.columns:
            return name
        found = by_normalized_name.get(str(name).strip().casefold())
        if found is not None:
            return found
    return None


_IMPL._first_present = _first_present

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep the patched helper visible to tests and exploratory callers.
globals()["_first_present"] = _first_present
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
