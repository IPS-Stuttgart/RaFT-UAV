"""Compatibility wrapper for robust branch-uncertainty probability CSV input.

The maintained implementation lives in the sibling
``candidate_branch_uncertainty.py`` module. This package preserves the public
import path while routing class-probability CSV reads through the shared
text-preserving reader. That keeps opaque sequence identifiers intact and
normalizes spreadsheet-padded headers before class context is attached.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad.class_probability_csv import read_class_probability_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_branch_uncertainty.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_branch_uncertainty_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load branch-aware candidate uncertainty implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


class _PandasCsvProxy:
    """Delegate pandas operations while hardening the CLI probability read."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if not args and kwargs.get("dtype") is str:
            return read_class_probability_csv(Path(path))
        return self._module.read_csv(path, *args, **kwargs)


_IMPL.pd = _PandasCsvProxy(pd)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
