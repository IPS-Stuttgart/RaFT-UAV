"""Compatibility fixes for Track 5 consensus template headers.

The maintained implementation lives in the sibling
``track5_estimate_consensus_ensemble.py`` module. This package preserves the
public import path while making template alias lookup insensitive to surrounding
whitespace and rejecting ambiguous alias columns before sequence/time alignment.
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


def _normalized_column_name(value: object) -> str:
    """Return the whitespace-insensitive, case-folded column name."""

    return str(value).strip().casefold()


def _first_present(rows: Any, names: tuple[str, ...]) -> Any | None:
    """Return the unique original column matching one of the supplied aliases."""

    aliases = {_normalized_column_name(name) for name in names}
    matching_columns = [
        column
        for column in rows.columns
        if _normalized_column_name(column) in aliases
    ]
    if len(matching_columns) > 1:
        rendered = ", ".join(repr(str(column)) for column in matching_columns)
        raise ValueError(
            "template contains ambiguous columns matching "
            f"{tuple(names)!r}: {rendered}"
        )
    if matching_columns:
        return matching_columns[0]
    return None


_IMPL._first_present = _first_present

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep the patched helpers visible to tests and exploratory callers.
globals()["_normalized_column_name"] = _normalized_column_name
globals()["_first_present"] = _first_present
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
