"""Compatibility wrapper for Track 5 submission ensembling.

The maintained implementation lives in the sibling
``track5_submission_ensemble.py`` module. This package keeps the public import
path while validating normalized ``Classification`` inputs with the same rules
as official Track 5 submissions, avoiding silent truncation of fractional class
ids.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import parse_official_classification_cell

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_submission_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_submission_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 submission ensemble implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _normalize_internal_submission_rows_with_class_validation(
    rows: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    """Normalize internal rows without coercing invalid class ids to integers."""

    frame = pd.DataFrame(rows).copy()
    lookup = _IMPL._normalized_column_lookup(frame)
    classification_column = _IMPL._normalized_classification_column(lookup)
    if classification_column is None:
        raise ValueError(f"{source_path} missing normalized Classification/classification column")
    out = pd.DataFrame(
        {
            "sequence_id": frame[lookup["sequence_id"]].astype(str),
            "time_s": pd.to_numeric(frame[lookup["time_s"]], errors="coerce"),
            "state_x_m": pd.to_numeric(frame[lookup["state_x_m"]], errors="coerce"),
            "state_y_m": pd.to_numeric(frame[lookup["state_y_m"]], errors="coerce"),
            "state_z_m": pd.to_numeric(frame[lookup["state_z_m"]], errors="coerce"),
            "Classification": frame[classification_column],
        }
    )
    finite = np.isfinite(
        out[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ).all(axis=1)
    out = out.loc[finite].copy()
    if out.empty:
        raise ValueError(f"{source_path} contains no finite normalized submission rows")
    out["Classification"] = [
        _parse_normalized_classification(value, source_path=source_path, row_index=row_index)
        for row_index, value in out["Classification"].items()
    ]
    return out.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _parse_normalized_classification(value: Any, *, source_path: Path, row_index: Any) -> int:
    try:
        return parse_official_classification_cell(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid Track 5 Classification in {source_path} at row {row_index}: {exc}"
        ) from exc


_IMPL._normalize_internal_submission_rows = (
    _normalize_internal_submission_rows_with_class_validation
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
