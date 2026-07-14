"""Compatibility wrapper preserving candidate-reservoir sequence text."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate reservoir implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_MAIN = _IMPL.main
_MAIN_LOCK = threading.RLock()


class _TextPreservingPandasProxy:
    """Delegate pandas operations while keeping CSV identifiers as text."""

    def __init__(self, pandas_module: Any) -> None:
        self._pandas_module = pandas_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pandas_module, name)

    def read_csv(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        kwargs.setdefault("dtype", str)
        kwargs.setdefault("keep_default_na", False)
        frame = self._pandas_module.read_csv(*args, **kwargs)
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame


def _read_sequence_text_csv(path: Path) -> pd.DataFrame:
    """Read CSV rows without coercing opaque sequence identifiers."""

    try:
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    except TypeError:
        rows = pd.read_csv(path, dtype=str)
    rows = rows.copy()
    rows.columns = [str(column).strip() for column in rows.columns]
    return rows


def load_candidate_inputs(specs: Sequence[str]) -> pd.DataFrame:
    """Load candidate CSV specs while preserving opaque sequence ids."""

    frames: list[pd.DataFrame] = []
    for spec in specs:
        branch, path = _IMPL._split_candidate_spec(str(spec))
        rows = _IMPL.normalize_candidate_columns(_read_sequence_text_csv(path))
        if rows.empty:
            continue
        _IMPL._validate_required_columns(rows, path)
        rows = rows.copy()
        if "source" not in rows.columns:
            rows["source"] = "unknown"
        if "track_id" not in rows.columns:
            rows["track_id"] = np.arange(len(rows), dtype=int).astype(str)
        if "candidate_branch" not in rows.columns:
            rows["candidate_branch"] = branch
        else:
            rows["candidate_branch"] = rows["candidate_branch"].fillna(branch).astype(str)
            rows.loc[rows["candidate_branch"].str.len() == 0, "candidate_branch"] = branch
        if "original_x_m" not in rows.columns:
            rows["original_x_m"] = pd.to_numeric(rows["x_m"], errors="coerce")
            rows["original_y_m"] = pd.to_numeric(rows["y_m"], errors="coerce")
            rows["original_z_m"] = pd.to_numeric(rows["z_m"], errors="coerce")
        rows["candidate_branch_input_path"] = str(path)
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=[*_IMPL._REQUIRED_COLUMNS, "source", "candidate_branch"])
    return pd.concat(frames, ignore_index=True)


def _load_candidate_specs(specs: list[str]) -> pd.DataFrame:
    candidates = load_candidate_inputs(specs)
    if candidates.empty:
        raise ValueError("at least one --candidate BRANCH=PATH entry is required")
    return candidates


def main(argv: list[str] | None = None) -> int:
    """Run the canonical CLI with a module-local text-preserving CSV reader."""

    with _MAIN_LOCK:
        original_impl_pd = _IMPL.pd
        _IMPL.pd = _TextPreservingPandasProxy(pd)
        try:
            return int(_ORIGINAL_MAIN(argv))
        finally:
            _IMPL.pd = original_impl_pd


_IMPL.load_candidate_inputs = load_candidate_inputs
_IMPL._load_candidate_specs = _load_candidate_specs
_IMPL.main = main

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["load_candidate_inputs"] = load_candidate_inputs
globals()["_load_candidate_specs"] = _load_candidate_specs
globals()["_read_sequence_text_csv"] = _read_sequence_text_csv
globals()["main"] = main

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
