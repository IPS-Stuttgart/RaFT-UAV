"""Compatibility fixes for MMUAD schema normalization.

The maintained implementation lives in the sibling ``schema.py`` module. This
package keeps the public ``raft_uav.mmuad.schema`` import path while filling
sparse canonical columns from complete aliases before defaults or finite-row
filters are applied.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "schema.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._schema_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load MMUAD schema implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def normalize_candidate_columns(
    frame: pd.DataFrame,
    *,
    default_sequence_id: str = "default",
    default_source: str = "candidate",
) -> pd.DataFrame:
    """Return a normalized candidate table with sparse aliases filled row-wise."""

    out = normalize_time_column_aliases(frame.copy(), target="time_s")
    out = _rename_aliases(out)
    out = _fill_sparse_alias_values(out, skip={"time_s"})
    if out.empty:
        return pd.DataFrame(columns=CANONICAL_CANDIDATE_COLUMNS)
    if "sequence_id" not in out.columns:
        out["sequence_id"] = default_sequence_id
    if "source" not in out.columns:
        out["source"] = default_source
    if "track_id" not in out.columns:
        out["track_id"] = np.nan
    if "std_xy_m" not in out.columns:
        out["std_xy_m"] = 10.0
    if "std_z_m" not in out.columns:
        out["std_z_m"] = out["std_xy_m"]
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    if "class_name" not in out.columns:
        out["class_name"] = "uav"
    missing_required = {"time_s", "x_m", "y_m", "z_m"}.difference(out.columns)
    if missing_required:
        raise ValueError(
            f"candidate table missing required columns: {sorted(missing_required)}; "
            f"available={list(out.columns)}"
        )
    for col in ("time_s", "x_m", "y_m", "z_m", "std_xy_m", "std_z_m", "confidence"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sequence_id"] = _normalize_sequence_id_values(
        out["sequence_id"],
        default_sequence_id=default_sequence_id,
    )
    out["source"] = _normalize_text_values(
        out["source"],
        default_text=default_source,
    )
    if "track_id" in out.columns:
        out["track_id"] = _normalize_optional_id_values(out["track_id"])
    out = out.loc[np.isfinite(out[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()
    return out.sort_values(["sequence_id", "time_s", "source"]).reset_index(drop=True)


def normalize_truth_columns(
    frame: pd.DataFrame,
    *,
    default_sequence_id: str = "default",
) -> pd.DataFrame:
    """Return a normalized truth table with sparse aliases filled row-wise."""

    out = normalize_time_column_aliases(frame.copy(), target="time_s")
    out = _rename_aliases(out)
    out = _fill_sparse_alias_values(out, skip={"time_s"})
    if out.empty:
        return pd.DataFrame(columns=CANONICAL_TRUTH_COLUMNS)
    if "sequence_id" not in out.columns:
        out["sequence_id"] = default_sequence_id
    for col in ("time_s", "x_m", "y_m", "z_m"):
        if col not in out.columns:
            raise ValueError(f"truth table missing {col!r}; available={list(out.columns)}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sequence_id"] = _normalize_sequence_id_values(
        out["sequence_id"],
        default_sequence_id=default_sequence_id,
    )
    out = out.loc[np.isfinite(out[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()
    return out.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _fill_sparse_alias_values(
    frame: pd.DataFrame,
    *,
    skip: Iterable[str] = (),
) -> pd.DataFrame:
    """Fill missing canonical values from aliases without overwriting present values."""

    out = frame.copy()
    skipped = {str(column) for column in skip}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in skipped:
            continue
        lower_to_original = _column_lookup(out.columns)
        canonical_column = lower_to_original.get(_column_key(canonical))
        if canonical_column is None:
            continue
        if canonical_column != canonical:
            out = out.rename(columns={canonical_column: canonical})
            canonical_column = canonical
            lower_to_original = _column_lookup(out.columns)
        combined = out[canonical_column].copy()
        for alias in aliases:
            alias_column = lower_to_original.get(_column_key(alias))
            if alias_column is None or alias_column == canonical_column:
                continue
            combined = combined.where(~_missing_like_mask(combined), out[alias_column])
        out[canonical] = combined
    return out


def _missing_like_mask(values: pd.Series) -> pd.Series:
    """Return rows whose scalar value should be treated as missing."""

    text = values.where(values.notna(), "").astype(str).str.strip().str.lower()
    return values.isna() | text.eq("") | text.isin({"nan", "none", "<na>"})


_IMPL.normalize_candidate_columns = normalize_candidate_columns
_IMPL.normalize_truth_columns = normalize_truth_columns
globals()["normalize_candidate_columns"] = normalize_candidate_columns
globals()["normalize_truth_columns"] = normalize_truth_columns

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
