"""Package wrapper that hardens Track 5 segment source-gate template handling.

The implementation lives in the sibling ``track5_segment_source_gate.py`` file.
This wrapper keeps the public import path while accepting spreadsheet-exported
template DataFrames with whitespace around alias headers and canonicalizing
opaque sequence IDs with the shared official Track 5 parser.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.mmuad.submission import parse_official_sequence_cell

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_segment_source_gate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_segment_source_gate_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 segment source-gate implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_TEMPLATE_ROWS_ATTR = "_raft_uav_original_normalize_template_rows"
_ORIGINAL_FIRST_PRESENT_ATTR = "_raft_uav_original_first_present"

if not hasattr(_IMPL, _ORIGINAL_NORMALIZE_TEMPLATE_ROWS_ATTR):
    setattr(
        _IMPL,
        _ORIGINAL_NORMALIZE_TEMPLATE_ROWS_ATTR,
        _IMPL._normalize_template_rows,
    )
if not hasattr(_IMPL, _ORIGINAL_FIRST_PRESENT_ATTR):
    setattr(_IMPL, _ORIGINAL_FIRST_PRESENT_ATTR, _IMPL._first_present)


def _official_sequence_text(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _first_present_with_stripped_headers(rows: Any, names: tuple[str, ...]) -> Any | None:
    lower = {str(column).strip().casefold(): column for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(str(name).casefold())
        if found is not None:
            return found
    return None


def _normalize_template_rows_with_official_sequence_ids(template: Any) -> Any:
    rows = _IMPL.pd.DataFrame(template).copy()
    sequence_column = _first_present_with_stripped_headers(
        rows,
        ("sequence_id", "Sequence", "sequence", "seq"),
    )
    time_column = _first_present_with_stripped_headers(
        rows,
        ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"),
    )
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = _IMPL.pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].map(_official_sequence_text),
            "time_s": _IMPL.pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & _IMPL.np.isfinite(out["time_s"].to_numpy(float))
    return (
        out.loc[finite]
        .drop_duplicates()
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


_IMPL._first_present = _first_present_with_stripped_headers
_IMPL._normalize_template_rows = _normalize_template_rows_with_official_sequence_ids

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

_first_present = _first_present_with_stripped_headers
_normalize_template_rows = _normalize_template_rows_with_official_sequence_ids
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
