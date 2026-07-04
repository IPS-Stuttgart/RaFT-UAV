"""Compatibility wrapper for MMUAD submission helpers."""

from __future__ import annotations

from typing import Any

from raft_uav.mmuad import _submission_impl as _impl

_parse_original = _impl.parse_official_classification_cell


def _parse_official_classification_cell_with_domain(value: Any) -> int:
    class_id = _parse_original(value)
    if class_id not in _impl.OFFICIAL_TRACK5_CLASS_IDS:
        raise ValueError(f"invalid official Track 5 class id: {class_id!r}")
    return class_id


_impl.parse_official_classification_cell = _parse_official_classification_cell_with_domain

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
parse_official_classification_cell = _parse_official_classification_cell_with_domain
