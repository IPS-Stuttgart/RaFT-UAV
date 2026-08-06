"""Reject ambiguous MMUAD evaluator headers before pandas can mangle them."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _normalized_header(column: Any) -> str:
    """Return the evaluator's case- and whitespace-insensitive header key."""

    return str(column).strip().lower()


def _validate_unique_headers(columns: Any) -> None:
    """Reject physical columns that collapse to the same normalized name."""

    physical_by_key: dict[str, list[str]] = {}
    for column in columns:
        key = _normalized_header(column)
        physical_by_key.setdefault(key, []).append(str(column))

    collisions = {
        key: physical
        for key, physical in physical_by_key.items()
        if len(physical) > 1
    }
    if collisions:
        details = ", ".join(
            f"{key!r}: {physical!r}"
            for key, physical in sorted(collisions.items())
        )
        raise ValueError(
            "MMUAD results contain ambiguous columns after trimming whitespace "
            f"and ignoring case: {details}"
        )


def install() -> None:
    """Install normalized collision validation at evaluator input boundaries."""

    evaluator = import_module("raft_uav.mmuad.evaluator")
    if getattr(evaluator, "_all_header_collision_patch_applied", False):
        return
    evaluator._validate_unique_official_track5_headers = _validate_unique_headers
    evaluator._all_header_collision_patch_applied = True
