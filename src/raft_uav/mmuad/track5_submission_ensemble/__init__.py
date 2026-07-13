"""Compatibility wrapper hardening normalized Track 5 submission inputs.

The maintained implementation lives in the sibling
``track5_submission_ensemble.py`` module. This package keeps the public import
path while rejecting malformed normalized rows instead of silently deleting or
truncating them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

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


def _invalid_row_summary(index: pd.Index, invalid: np.ndarray) -> str:
    positions = np.flatnonzero(np.asarray(invalid, dtype=bool))
    labels = [repr(index[int(position)]) for position in positions[:5]]
    suffix = ", ..." if len(positions) > 5 else ""
    return ", ".join(labels) + suffix


def _raise_invalid_rows(
    *,
    source_path: Path,
    index: pd.Index,
    invalid: np.ndarray,
    field: str,
) -> None:
    count = int(np.count_nonzero(invalid))
    sample = _invalid_row_summary(index, invalid)
    raise ValueError(
        f"{source_path} contains {count} invalid normalized {field} row(s) "
        f"at index/indices {sample}"
    )


def _normalize_internal_submission_rows(
    rows: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    """Normalize internal rows without silently dropping or truncating values."""

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
            "Classification": pd.to_numeric(frame[classification_column], errors="coerce"),
        },
        index=frame.index,
    )
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"):
        invalid = ~np.isfinite(out[column].to_numpy(dtype=float))
        if invalid.any():
            _raise_invalid_rows(
                source_path=source_path,
                index=out.index,
                invalid=invalid,
                field=column,
            )
    try:
        labels = _IMPL._predicted_class_labels(out["Classification"])
        out["Classification"] = labels.map(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_path} contains invalid normalized Classification values"
        ) from exc
    return out.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


_IMPL._normalize_internal_submission_rows = _normalize_internal_submission_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep patched helpers importable for focused regressions.
globals()["_invalid_row_summary"] = _invalid_row_summary
globals()["_raise_invalid_rows"] = _raise_invalid_rows
globals()["_normalize_internal_submission_rows"] = _normalize_internal_submission_rows

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
