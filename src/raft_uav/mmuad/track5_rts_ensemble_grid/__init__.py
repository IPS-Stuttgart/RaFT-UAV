"""Package wrapper that preserves opaque sequence IDs in RTS grid inputs.

The implementation lives in the sibling ``track5_rts_ensemble_grid.py`` file.
This wrapper keeps the public import path while routing estimate CSV reads through
the shared text-preserving reader so identifiers such as ``001`` are not coerced
to integers before template matching.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_rts_ensemble_grid.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_rts_ensemble_grid_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 RTS ensemble grid implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


class _PandasEstimateCsvProxy:
    """Delegate pandas operations while preserving estimate identifier columns."""

    def __init__(self, pandas_module: Any) -> None:
        self._pandas = pandas_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pandas, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args or kwargs:
            return self._pandas.read_csv(path, *args, **kwargs)
        return read_estimate_csv(Path(path))


_IMPL.pd = _PandasEstimateCsvProxy(pd)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
