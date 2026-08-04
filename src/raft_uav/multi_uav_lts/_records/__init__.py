"""Strict parsing overrides for Multi-UAV LTS records.

The maintained implementation lives in the sibling ``_records.py`` module. This
package preserves the public import path while enforcing the same confidence
domain used by submission validation before parsed rows reach scoring or
fixed-population post-processing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_IMPL_PATH = Path(__file__).resolve().parent.parent / "_records.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.multi_uav_lts._records_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError(f"cannot load Multi-UAV LTS records implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_PARSE_DETECTION_TEXT = _IMPL.parse_detection_text


def parse_detection_text(text: str, *, source: str):
    """Parse detections and reject confidence values outside ``[-1, 1]``."""

    rows = _ORIGINAL_PARSE_DETECTION_TEXT(text, source=source)
    line_numbers = [
        line_number
        for line_number, raw in enumerate(text.splitlines(), start=1)
        if raw.strip()
    ]
    for line_number, row in zip(line_numbers, rows, strict=True):
        if not -1.0 <= row.confidence <= 1.0:
            raise ValueError(
                f"{source}:{line_number}: confidence must be in [-1, 1]"
            )
    return rows


_IMPL.parse_detection_text = parse_detection_text

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["parse_detection_text"] = parse_detection_text

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
