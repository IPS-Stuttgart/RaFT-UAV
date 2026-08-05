"""Reject Boolean official Track 5 sequence identifiers."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np


_ORIGINAL_PARSE_ATTR = "_raft_uav_original_parse_official_sequence_cell"


def _submission_modules() -> list[ModuleType]:
    """Return the public, compatibility, and core submission modules."""

    from raft_uav.mmuad import submission

    modules = [submission]
    legacy = getattr(submission, "_IMPL", None)
    if isinstance(legacy, ModuleType):
        modules.append(legacy)
        core = getattr(legacy, "_impl", None)
        if isinstance(core, ModuleType):
            modules.append(core)
    return modules


def install() -> None:
    """Reject Boolean values at every shared official-sequence parser boundary."""

    modules = _submission_modules()
    owner = modules[-1]
    if not hasattr(owner, _ORIGINAL_PARSE_ATTR):
        setattr(owner, _ORIGINAL_PARSE_ATTR, owner.parse_official_sequence_cell)
    original = getattr(owner, _ORIGINAL_PARSE_ATTR)

    def parse_official_sequence_cell(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "official MMUAD Sequence values must be identifiers, not booleans"
            )
        return original(value)

    for module in modules:
        module.parse_official_sequence_cell = parse_official_sequence_cell
