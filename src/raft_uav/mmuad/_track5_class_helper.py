"""Track 5 class helpers."""

from __future__ import annotations

from typing import Any

from raft_uav.mmuad import submission as _submission


def install() -> None:
    original = _submission.parse_official_classification_cell
    if getattr(original, "_raft_uav_class_domain", False):
        return

    def parse_with_domain(value: Any) -> int:
        class_id = original(value)
        if class_id not in _submission.OFFICIAL_TRACK5_CLASS_IDS:
            raise ValueError(f"invalid official Track 5 class id: {class_id!r}")
        return class_id

    setattr(parse_with_domain, "_raft_uav_class_domain", True)
    _submission.parse_official_classification_cell = parse_with_domain
