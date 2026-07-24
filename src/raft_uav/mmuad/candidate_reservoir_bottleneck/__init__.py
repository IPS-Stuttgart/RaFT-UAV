"""Compatibility fixes for reservoir bottleneck diagnostics.

The maintained implementation lives in the sibling
``candidate_reservoir_bottleneck.py`` module. This package preserves the
public import path while keeping annotations and summary row selection
positional, rejecting non-finite metric inputs, and writing strict JSON
summaries.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float as _shared_optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_bottleneck.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_bottleneck_impl",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import guard
    raise ImportError(f"could not load reservoir bottleneck implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _optional_float(value: Any) -> float | None:
    """Return a finite scalar float or ``None`` for invalid metrics."""

    return _shared_optional_float(value)


def _jsonable(value: Any) -> Any:
    """Recursively normalize NumPy and non-finite values for strict JSON."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def annotate_gap_table(
    gap_rows: pd.DataFrame,
    *,
    config: Any = None,
) -> pd.DataFrame:
    """Append annotations by row position rather than index-label alignment."""

    rows = pd.DataFrame(gap_rows).copy()
    if rows.empty:
        return rows.assign(
            primary_bottleneck=pd.Series(dtype=str),
            recommended_action=pd.Series(dtype=str),
        )
    annotations = pd.DataFrame.from_records(
        [
            _IMPL.classify_gap_row(record, config=config)
            for record in rows.to_dict(orient="records")
        ]
    )
    add_columns = [
        column
        for column in annotations.columns
        if column not in rows.columns or column in _IMPL._OVERWRITE_COLUMNS
    ]
    for column in add_columns:
        rows[column] = annotations[column].to_numpy()
    return rows


def _max_record(rows: pd.DataFrame, column: str) -> dict[str, Any]:
    """Return the maximum-valued row without coercing its index label."""

    if rows.empty or column not in rows.columns:
        return {}
    values = pd.to_numeric(rows[column], errors="coerce")
    if values.dropna().empty:
        return {}
    numeric_values = values.to_numpy(dtype=float, na_value=np.nan)
    position = int(np.nanargmax(numeric_values))
    return _jsonable(rows.iloc[position].to_dict())


def write_bottleneck_outputs(
    annotated: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    config: Any = None,
) -> dict[str, Path]:
    """Write annotated rows and a standards-compliant optional JSON summary."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_csv, index=False)
    paths = {"output_csv": output_csv}
    if summary_json is not None:
        summary_json = Path(summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _IMPL.build_bottleneck_summary(
            annotated,
            config=config or _IMPL.BottleneckConfig(),
        )
        summary_json.write_text(
            json.dumps(_jsonable(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        paths["summary_json"] = summary_json
    return paths


_IMPL._optional_float = _optional_float
_IMPL._jsonable = _jsonable
_IMPL.annotate_gap_table = annotate_gap_table
_IMPL._max_record = _max_record
_IMPL.write_bottleneck_outputs = write_bottleneck_outputs

for _name, _value in vars(_IMPL).items():
    if _name not in {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
    }:
        globals()[_name] = _value

globals()["_optional_float"] = _optional_float
globals()["_jsonable"] = _jsonable
globals()["annotate_gap_table"] = annotate_gap_table
globals()["_max_record"] = _max_record
globals()["write_bottleneck_outputs"] = write_bottleneck_outputs

__doc__ = _IMPL.__doc__
__all__ = [name for name in vars(_IMPL) if not name.startswith("_")]
