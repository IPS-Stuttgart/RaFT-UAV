"""Compatibility wrapper preserving exact paper-strict diagnostic identifiers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.numeric import optional_int as _optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "paper_strict.py"
_IMPL_NAME = f"{__name__.rsplit('.', 1)[0]}._paper_strict_legacy"
_SPEC = importlib.util.spec_from_file_location(_IMPL_NAME, _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load paper-strict implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRACKING_RECORD = _IMPL._tracking_record


def _tracking_record(
    measurement: Any,
    tracker: Any,
    diagnostics: Any,
    *,
    association_mode: str,
    selected_row: Any | None = None,
) -> dict[str, object]:
    """Build a tracking record without routing integer metadata through float."""

    record = _ORIGINAL_TRACKING_RECORD(
        measurement,
        tracker,
        diagnostics,
        association_mode=association_mode,
        selected_row=None,
    )
    if selected_row is None:
        return record

    for field in ("track_id", "frame_index"):
        if field not in selected_row.index:
            continue
        value = _optional_int(selected_row[field])
        if value is not None:
            record[field] = value
    return record


_IMPL._tracking_record = _tracking_record

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_tracking_record"] = _tracking_record
globals()["_optional_int"] = _optional_int

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
