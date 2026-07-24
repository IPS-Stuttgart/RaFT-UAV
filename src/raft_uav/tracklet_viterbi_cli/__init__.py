"""Compatibility wrapper for the canonical tracklet-Viterbi CLI.

The maintained implementation lives in the sibling ``tracklet_viterbi_cli.py``
module. This package preserves the public import path while ensuring that a
negative range-gate environment override is rejected instead of silently
turning the gate off.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_viterbi_cli.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav._tracklet_viterbi_cli_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load canonical tracklet CLI implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ENV_OPTIONAL_POSITIVE_FLOAT = _IMPL._env_optional_positive_float


def _env_optional_positive_float(name: str, default: float | None) -> float | None:
    """Parse an optional positive environment value without accepting negatives.

    Zero remains the explicit sentinel for disabling an optional gate. Negative
    values are configuration errors and must not be conflated with that sentinel.
    """

    value = os.environ.get(name)
    if value is not None and value != "":
        parsed = float(value)
        if parsed < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    return _ORIGINAL_ENV_OPTIONAL_POSITIVE_FLOAT(name, default)


_IMPL._env_optional_positive_float = _env_optional_positive_float

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_env_optional_positive_float"] = _env_optional_positive_float

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
