"""Compatibility layer with strict runtime scalar-control validation.

The maintained implementation lives in the sibling ``runtime_cli_config.py``
module. This package preserves the public import path while rejecting malformed
integer and floating-point controls before ``int(...)`` or ``float(...)`` can
truncate, unwrap, or otherwise coerce them.
"""

from __future__ import annotations

import importlib.util
import numbers
import sys
from collections.abc import Iterable
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


def _runtime_option_from_token(token: str) -> str | None:
    """Resolve an exact or uniquely abbreviated runtime long option."""

    option = token.partition("=")[0]
    if option == "--" or not option.startswith("--"):
        return None
    if option in _IMPL._RUNTIME_FLAG_ENV_NAMES:
        return option
    matches = [
        candidate
        for candidate in _IMPL._RUNTIME_FLAG_ENV_NAMES
        if candidate.startswith(option)
    ]
    return matches[0] if len(matches) == 1 else None


def runtime_environment_names_from_argv(argv: Iterable[str]) -> set[str]:
    """Return runtime env vars explicitly controlled by *argv* flags."""

    names: set[str] = set()
    for token in argv:
        if token == "--":
            break
        option = _runtime_option_from_token(token)
        if option is not None:
            names.update(_IMPL._RUNTIME_FLAG_ENV_NAMES.get(option, ()))
    return names


def _runtime_passthrough_arguments(argv: Iterable[str]) -> list[str]:
    """Restore runtime options that are also consumed by the base CLI."""

    restored: list[str] = []
    tokens = list(argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            break
        _, separator, value = token.partition("=")
        option = _runtime_option_from_token(token)
        if option in _IMPL._RUNTIME_PASSTHROUGH_FLAGS:
            restored.append(f"{option}={value}" if separator else option)
            if not separator and index + 1 < len(tokens):
                index += 1
                restored.append(tokens[index])
        index += 1
    return restored


def _invalid_float(name: str) -> ValueError:
    return ValueError(f"{name} must be a finite real scalar")


def _finite_float(value: object, name: str) -> float:
    """Return a finite real scalar without Boolean or array coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise _invalid_float(name)
    if np.ma.isMaskedArray(value):
        if bool(np.ma.getmaskarray(value).any()):
            raise _invalid_float(name)
        value = np.ma.getdata(value)

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise _invalid_float(name) from exc
    if array.ndim != 0 or np.iscomplexobj(array):
        raise _invalid_float(name)

    scalar = array.item()
    if np.ma.is_masked(scalar) or isinstance(
        scalar,
        (bool, np.bool_, complex, np.complexfloating),
    ):
        raise _invalid_float(name)
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid_float(name) from exc
    if not np.isfinite(number):
        raise _invalid_float(name)
    return number


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


_IMPL._runtime_option_from_token = _runtime_option_from_token
_IMPL.runtime_environment_names_from_argv = runtime_environment_names_from_argv
_IMPL._runtime_passthrough_arguments = _runtime_passthrough_arguments
_IMPL._finite_float = _finite_float
_IMPL._positive_int = _positive_int
_IMPL._nonnegative_int = _nonnegative_int

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_runtime_option_from_token"] = _runtime_option_from_token
globals()["runtime_environment_names_from_argv"] = runtime_environment_names_from_argv
globals()["_runtime_passthrough_arguments"] = _runtime_passthrough_arguments
globals()["_finite_float"] = _finite_float
globals()["_positive_int"] = _positive_int
globals()["_nonnegative_int"] = _nonnegative_int

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
