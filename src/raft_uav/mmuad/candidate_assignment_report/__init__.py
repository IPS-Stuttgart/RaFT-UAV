"""Compatibility package for candidate-assignment report CSV ingestion.

The maintained implementation lives in the sibling
``candidate_assignment_report.py`` module. This package preserves the public
import path while preventing pandas from coercing opaque numeric-looking
``sequence_id`` values in assignment CSV files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from typing import Any

from raft_uav.mmuad.estimate_csv import read_estimate_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_assignment_report.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_assignment_report_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        "cannot load candidate-assignment report implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_MAIN = _IMPL.main
_MAIN_LOCK = threading.RLock()


class _AssignmentReportPandasProxy:
    """Proxy pandas while preserving assignment CSV identifier columns."""

    def __init__(self, pandas_module: Any) -> None:
        self._pandas_module = pandas_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pandas_module, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return self._pandas_module.read_csv(path, *args, **kwargs)
        return read_estimate_csv(Path(path))


def main(argv: list[str] | None = None) -> int:
    """Run the report CLI with sequence-ID-preserving assignment CSV parsing."""

    with _MAIN_LOCK:
        pandas_module = _IMPL.pd
        _IMPL.pd = _AssignmentReportPandasProxy(pandas_module)
        try:
            return int(_LEGACY_MAIN(argv))
        finally:
            _IMPL.pd = pandas_module


_IMPL.main = main

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["main"] = main

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
