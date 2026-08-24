"""Compatibility guard for candidate-pool ablation label collisions.

The maintained implementation lives in the sibling
``candidate_pool_branch_ablation.py`` module. This package preserves the public
import path while rejecting group values that would generate the same ablation
pool label and silently overwrite one another.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_pool_branch_ablation.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pool_branch_ablation_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        "cannot load candidate-pool branch ablation implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_POOL_BRANCH_ABLATION_POOLS = (
    _IMPL.build_candidate_pool_branch_ablation_pools
)


def _validate_unique_group_slugs(rows: pd.DataFrame, *, group_column: str) -> None:
    """Reject distinct group values that normalize to the same pool-label slug."""

    values_by_slug: dict[str, list[str]] = {}
    group_values = sorted(rows[group_column].dropna().astype(str).unique().tolist())
    for group_value in group_values:
        values_by_slug.setdefault(_IMPL._slug(group_value), []).append(group_value)

    collisions = {
        slug: values
        for slug, values in values_by_slug.items()
        if len(values) > 1
    }
    if not collisions:
        return

    details = "; ".join(
        f"{', '.join(repr(value) for value in values)} -> {slug!r}"
        for slug, values in sorted(collisions.items())
    )
    raise ValueError(
        f"{group_column} values collide after pool-label normalization: {details}"
    )


def build_candidate_pool_branch_ablation_pools(
    candidates: pd.DataFrame,
    *,
    group_column: str = "candidate_branch",
    include_full_pool: bool = True,
    include_leave_one_out: bool = True,
    include_only_one: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build ablation pools after validating generated pool-label identity."""

    rows = pd.DataFrame(candidates).copy()
    if not rows.empty and (include_leave_one_out or include_only_one):
        if group_column not in rows.columns:
            rows[group_column] = "unknown"
        rows[group_column] = _IMPL._clean_label_series(rows[group_column])
        _validate_unique_group_slugs(rows, group_column=group_column)
    return _ORIGINAL_BUILD_CANDIDATE_POOL_BRANCH_ABLATION_POOLS(
        rows,
        group_column=group_column,
        include_full_pool=include_full_pool,
        include_leave_one_out=include_leave_one_out,
        include_only_one=include_only_one,
    )


_IMPL.build_candidate_pool_branch_ablation_pools = (
    build_candidate_pool_branch_ablation_pools
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["build_candidate_pool_branch_ablation_pools"] = (
    build_candidate_pool_branch_ablation_pools
)
globals()["_validate_unique_group_slugs"] = _validate_unique_group_slugs
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
