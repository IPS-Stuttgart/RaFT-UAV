"""Compatibility fixes for branch-aware MMUAD candidate uncertainty.

The maintained implementation lives in the sibling
``candidate_branch_uncertainty.py`` module. This package preserves the public
import path while routing class-probability CSV reads through the shared
text-preserving reader and canonicalizing source/branch identities for
model-feature grouping without rewriting the labels returned to callers.
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
_ORIGINAL_ATTACH_BRANCH_UNCERTAINTY_CONTEXT = (
    _IMPL.attach_branch_uncertainty_context
)


def _canonical_identity_labels(
    values: Any,
    *,
    default: str = "candidate",
) -> pd.Series:
    """Normalize categorical labels for identity comparisons only."""

    labels = pd.Series(values, copy=False)
    normalized = labels.fillna("").astype(str).str.strip().str.casefold()
    return normalized.where(normalized.str.len() > 0, str(default))


def attach_branch_uncertainty_context(
    candidates: Any,
    *,
    score_columns: Any = _IMPL.DEFAULT_SCORE_COLUMNS,
):
    """Attach branch features after canonicalizing grouping identities."""

    rows = _IMPL._candidate_rows(candidates)
    if rows.empty:
        return _ORIGINAL_ATTACH_BRANCH_UNCERTAINTY_CONTEXT(
            rows,
            score_columns=score_columns,
        )

    visible_branch = _IMPL._branch_values(rows).reset_index(drop=True)
    visible_source = (
        rows.get("source", pd.Series("candidate", index=rows.index))
        .fillna("candidate")
        .astype(str)
        .reset_index(drop=True)
    )

    identity_rows = rows.reset_index(drop=True).copy()
    identity_rows["candidate_branch"] = _canonical_identity_labels(
        visible_branch,
    ).to_numpy()
    identity_rows["source"] = _canonical_identity_labels(
        visible_source,
    ).to_numpy()

    contextual = _ORIGINAL_ATTACH_BRANCH_UNCERTAINTY_CONTEXT(
        identity_rows,
        score_columns=score_columns,
    )
    output = contextual.rows.copy().reset_index(drop=True)
    output["candidate_branch"] = visible_branch.to_numpy()
    output["source"] = visible_source.to_numpy()
    return _IMPL.CandidateFrame(_IMPL.normalize_candidate_columns(output))


_IMPL.attach_branch_uncertainty_context = attach_branch_uncertainty_context

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_canonical_identity_labels"] = _canonical_identity_labels
globals()["attach_branch_uncertainty_context"] = attach_branch_uncertainty_context

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
