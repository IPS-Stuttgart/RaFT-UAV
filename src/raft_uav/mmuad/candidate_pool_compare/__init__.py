"""Compatibility package that emits strict candidate-pool comparison JSON.

The maintained implementation lives in the sibling ``candidate_pool_compare.py``
module. This package preserves the public import path while normalizing missing
and non-finite pandas/NumPy values to JSON ``null``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_pool_compare.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pool_compare_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate-pool comparison implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _jsonable(value: Any) -> Any:
    """Return a strict-JSON-compatible representation of one value."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_candidate_pool_compare_outputs(
    *,
    output_dir: Path,
    frame_rows: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    by_sequence: pd.DataFrame,
    by_reference_branch: pd.DataFrame,
) -> dict[str, str]:
    """Write comparison artifacts with standards-compliant JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_csv": output_dir / "mmuad_candidate_pool_compare_frames.csv",
        "pooled_csv": output_dir / "mmuad_candidate_pool_compare_pooled.csv",
        "by_sequence_csv": output_dir / "mmuad_candidate_pool_compare_by_sequence.csv",
        "by_reference_branch_csv": output_dir
        / "mmuad_candidate_pool_compare_by_reference_branch.csv",
        "summary_json": output_dir / "mmuad_candidate_pool_compare_summary.json",
    }
    frame_rows.to_csv(paths["frame_csv"], index=False)
    pooled_summary.to_csv(paths["pooled_csv"], index=False)
    by_sequence.to_csv(paths["by_sequence_csv"], index=False)
    by_reference_branch.to_csv(paths["by_reference_branch_csv"], index=False)
    summary = {
        "pooled": pooled_summary.to_dict(orient="records"),
        "by_sequence": by_sequence.to_dict(orient="records"),
        "by_reference_branch": by_reference_branch.to_dict(orient="records"),
    }
    paths["summary_json"].write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


_IMPL.write_candidate_pool_compare_outputs = write_candidate_pool_compare_outputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
