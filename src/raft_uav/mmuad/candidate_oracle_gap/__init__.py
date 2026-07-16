"""Compatibility package for validated MMUAD candidate oracle-gap diagnostics."""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_gap.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_gap_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy candidate oracle-gap module from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP = _IMPL.build_candidate_oracle_gap


def _normalize_max_time_delta_s(value: Any) -> float | None:
    if value is None:
        return None
    normalized = optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(
            "max_time_delta_s must be a nonnegative finite scalar or None"
        )
    return normalized


@wraps(_ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP)
def build_candidate_oracle_gap(
    candidates: Any,
    selected: Any,
    truth: Any,
    *,
    max_time_delta_s: float | None = 0.5,
) -> Any:
    """Build oracle-gap rows after validating the nearest-time gate."""

    return _ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP(
        candidates,
        selected,
        truth,
        max_time_delta_s=_normalize_max_time_delta_s(max_time_delta_s),
    )


_IMPL.build_candidate_oracle_gap = build_candidate_oracle_gap

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

globals()["_normalize_max_time_delta_s"] = _normalize_max_time_delta_s
__doc__ = _IMPL.__doc__
__all__ = [_name for _name in dir(_IMPL) if not _name.startswith("__")]
