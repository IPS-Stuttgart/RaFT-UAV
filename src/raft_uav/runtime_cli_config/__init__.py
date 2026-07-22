"""Compatibility layer with strict runtime integer-control validation.

The maintained implementation lives in the sibling ``runtime_cli_config.py``
module. This package preserves the public import path while rejecting malformed
integer controls before they can be truncated by ``int(...)``.
"""

from __future__ import annotations

import importlib.util
import numbers
import sys
from pathlib import Path

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


def _invalid_integer(name: str, qualifier: str) -> ValueError:
    return ValueError(f"{name} must be a {qualifier} integer")


def _validated_integer(
    value: object,
    name: str,
    *,
    minimum: int,
    qualifier: str,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise _invalid_integer(name, qualifier)
    if np.ma.isMaskedArray(value):
        if bool(np.ma.getmaskarray(value).any()):
            raise _invalid_integer(name, qualifier)
        value = np.ma.getdata(value)

    array = np.asarray(value)
    if array.ndim != 0:
        raise _invalid_integer(name, qualifier)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise _invalid_integer(name, qualifier)

    if isinstance(scalar, numbers.Integral):
        number = int(scalar)
    elif isinstance(scalar, str):
        try:
            number = int(scalar.strip())
        except (TypeError, ValueError, OverflowError):
            raise _invalid_integer(name, qualifier) from None
    else:
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError):
            raise _invalid_integer(name, qualifier) from None
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise _invalid_integer(name, qualifier)
        number = int(numeric)

    if number < minimum:
        raise _invalid_integer(name, qualifier)
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
