"""Compatibility wrapper for per-sequence candidate-mixture restarts.

The maintained implementation lives in the sibling
``candidate_mixture_map_sequence_multistart.py`` module. This package preserves
its public import path, makes sequence-less external initial trajectories apply
to every candidate sequence, and measures branch restart eligibility within each
sequence instead of pooling coverage over the complete input batch.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_map_sequence_multistart.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_sequence_multistart_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load candidate-mixture sequence multi-start implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RUN_SEQUENCE_MULTISTART = _IMPL.run_sequence_multistart_candidate_mixture_map
_ORIGINAL_MULTISTART = _IMPL.multistart
_ORIGINAL_BUILD_INITIALIZATIONS = (
    _ORIGINAL_MULTISTART.build_candidate_mixture_initializations
)
_SEQUENCE_ALIASES = (
    "sequence_id",
    "Sequence",
    "sequence",
    "seq",
    "scene",
    "scene_id",
    "clip",
    "clip_id",
)


globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _expand_sequence_less_external_initialization(
    candidates: pd.DataFrame,
    external_initial_estimates: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Canonicalize an explicit sequence column or replicate a default trajectory."""

    if external_initial_estimates is None:
        return None
    rows = pd.DataFrame(external_initial_estimates).copy()
    rows.columns = [str(column).strip() for column in rows.columns]
    if rows.empty:
        return rows

    sequence_column = _first_present_column(rows, _SEQUENCE_ALIASES)
    if sequence_column is not None:
        rows["sequence_id"] = _sequence_id_text(rows[sequence_column])
        return rows

    candidate_rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    sequence_ids = sorted(
        candidate_rows["sequence_id"].astype(str).drop_duplicates().tolist()
    )
    if not sequence_ids:
        return rows

    parts: list[pd.DataFrame] = []
    for sequence_id in sequence_ids:
        part = rows.copy()
        part.insert(0, "sequence_id", str(sequence_id))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_sequence_candidate_mixture_initializations(
    candidates: pd.DataFrame,
    *,
    mixture_config: Any | None = None,
    multistart_config: Any | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Build restarts using branch coverage measured within each sequence."""

    mixture_config = mixture_config or _IMPL.core.CandidateMixtureMapConfig()
    multistart_config = (
        multistart_config
        or _ORIGINAL_MULTISTART.CandidateMixtureMultiStartConfig()
    )
    validator = getattr(_ORIGINAL_MULTISTART, "_validate_multistart_config", None)
    if validator is not None:
        validator(multistart_config)

    external = _expand_sequence_less_external_initialization(
        candidates,
        external_initial_estimates,
    )
    if not bool(multistart_config.include_branch_starts):
        return _ORIGINAL_BUILD_INITIALIZATIONS(
            candidates,
            mixture_config=mixture_config,
            multistart_config=multistart_config,
            external_initial_estimates=external,
        )

    exhaustive_config = replace(
        multistart_config,
        max_branch_starts=0,
        min_branch_frame_fraction=0.0,
    )
    starts = _ORIGINAL_BUILD_INITIALIZATIONS(
        candidates,
        mixture_config=mixture_config,
        multistart_config=exhaustive_config,
        external_initial_estimates=external,
    )
    eligible = set(
        _eligible_sequence_branches(
            candidates,
            branch_column=str(multistart_config.branch_column),
            config=multistart_config,
        )
    )
    return {
        name: rows
        for name, rows in starts.items()
        if not name.startswith("branch:")
        or name.removeprefix("branch:") in eligible
    }


def run_sequence_multistart_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: Any | None = None,
    multistart_config: Any | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> Any:
    """Run per-sequence multi-start with external initialization IDs made usable."""

    external = _expand_sequence_less_external_initialization(
        candidates,
        external_initial_estimates,
    )
    return _ORIGINAL_RUN_SEQUENCE_MULTISTART(
        candidates,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external,
        truth=truth,
    )


def _eligible_sequence_branches(
    candidates: pd.DataFrame,
    *,
    branch_column: str,
    config: Any,
) -> list[str]:
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return []
    rows = rows.reset_index(drop=True)
    if branch_column not in rows.columns:
        rows[branch_column] = (
            rows["source"].fillna("unknown").astype(str)
            if "source" in rows.columns
            else "unknown"
        )
    rows[branch_column] = rows[branch_column].fillna("unknown").astype(str)

    global_presence = (
        rows[["sequence_id", "time_s", branch_column]]
        .drop_duplicates()
        .groupby(branch_column, dropna=False)
        .size()
    )
    global_row_count = rows.groupby(branch_column, dropna=False).size()
    branch_stats: dict[str, dict[str, float]] = {}
    minimum_fraction = float(config.min_branch_frame_fraction)

    for _, sequence_rows in rows.groupby("sequence_id", sort=True):
        frame_count = int(sequence_rows["time_s"].nunique(dropna=True))
        if frame_count <= 0:
            continue
        minimum = max(1, int(np.ceil(minimum_fraction * frame_count)))
        presence = (
            sequence_rows[["time_s", branch_column]]
            .drop_duplicates()
            .groupby(branch_column, dropna=False)
            .size()
        )
        for branch, count in presence.items():
            count = int(count)
            if count < minimum:
                continue
            name = str(branch)
            local_fraction = float(count / frame_count)
            stats = branch_stats.setdefault(
                name,
                {"max_local_fraction": 0.0, "max_local_count": 0.0},
            )
            stats["max_local_fraction"] = max(
                stats["max_local_fraction"],
                local_fraction,
            )
            stats["max_local_count"] = max(
                stats["max_local_count"],
                float(count),
            )

    branches = list(branch_stats)
    branches.sort(
        key=lambda branch: (
            -branch_stats[branch]["max_local_fraction"],
            -branch_stats[branch]["max_local_count"],
            -int(global_presence.get(branch, 0)),
            -int(global_row_count.get(branch, 0)),
            branch,
        )
    )
    limit = int(config.max_branch_starts)
    return branches[:limit] if limit > 0 else branches


def _first_present_column(rows: pd.DataFrame, aliases: tuple[str, ...]) -> Any | None:
    by_normalized_name = {
        str(column).strip().casefold(): column for column in rows.columns
    }
    for alias in aliases:
        if alias in rows.columns:
            return alias
        found = by_normalized_name.get(str(alias).casefold())
        if found is not None:
            return found
    return None


def _sequence_id_text(values: pd.Series) -> pd.Series:
    """Return stripped sequence IDs while preserving opaque text such as ``001``."""

    return values.where(values.notna(), "").astype(str).str.strip()


class _SequenceAwareMultistartProxy:
    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def build_candidate_mixture_initializations(
        self,
        candidates: pd.DataFrame,
        *,
        mixture_config: Any | None = None,
        multistart_config: Any | None = None,
        external_initial_estimates: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame | None]:
        return build_sequence_candidate_mixture_initializations(
            candidates,
            mixture_config=mixture_config,
            multistart_config=multistart_config,
            external_initial_estimates=external_initial_estimates,
        )


_SEQUENCE_AWARE_MULTISTART = _SequenceAwareMultistartProxy(_ORIGINAL_MULTISTART)
_IMPL.multistart = _SEQUENCE_AWARE_MULTISTART
_IMPL.run_sequence_multistart_candidate_mixture_map = (
    run_sequence_multistart_candidate_mixture_map
)
globals()["multistart"] = _SEQUENCE_AWARE_MULTISTART
globals()["build_sequence_candidate_mixture_initializations"] = (
    build_sequence_candidate_mixture_initializations
)
globals()["run_sequence_multistart_candidate_mixture_map"] = (
    run_sequence_multistart_candidate_mixture_map
)
globals()["_expand_sequence_less_external_initialization"] = (
    _expand_sequence_less_external_initialization
)
globals()["_eligible_sequence_branches"] = _eligible_sequence_branches
globals()["_first_present_column"] = _first_present_column
globals()["_sequence_id_text"] = _sequence_id_text

__doc__ = _IMPL.__doc__
__all__ = sorted(
    {
        *[
            name
            for name in dir(_IMPL)
            if not (name.startswith("__") and name.endswith("__"))
        ],
        "build_sequence_candidate_mixture_initializations",
        "run_sequence_multistart_candidate_mixture_map",
    }
)
