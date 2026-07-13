"""Compatibility wrapper with strict runtime integer-control validation.

The maintained implementation lives in the sibling ``runtime_cli_config.py``
module. This package preserves the public import path while rejecting malformed
integer controls before they can be truncated by ``int(...)``.
"""

from __future__ import annotations

import importlib.util
import numbers
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "runtime_cli_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav._runtime_cli_config_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load runtime CLI configuration from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _validated_integer(value: object, name: str, *, minimum: int, qualifier: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a {qualifier} integer")
    if isinstance(value, numbers.Integral):
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value.strip())
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a {qualifier} integer") from None
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a {qualifier} integer") from None
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a {qualifier} integer")
        number = int(numeric)
    if number < minimum:
        raise ValueError(f"{name} must be a {qualifier} integer")
    return number


def _positive_int(value: object, name: str) -> int:
    return _validated_integer(value, name, minimum=1, qualifier="positive")


def _nonnegative_int(value: object, name: str) -> int:
    return _validated_integer(value, name, minimum=0, qualifier="nonnegative")


_IMPL._positive_int = _positive_int
_IMPL._nonnegative_int = _nonnegative_int

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_positive_int"] = _positive_int
globals()["_nonnegative_int"] = _nonnegative_int

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
