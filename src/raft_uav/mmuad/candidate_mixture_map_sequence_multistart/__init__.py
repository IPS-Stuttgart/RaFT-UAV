"""Compatibility wrapper for reusable per-sequence external initializations.

The maintained implementation lives in the sibling
``candidate_mixture_map_sequence_multistart.py`` module. This package preserves
its public import path while making sequence-less external initial trajectories
apply to every candidate sequence instead of being normalized to an unused
``default`` identifier.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_map_sequence_multistart.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_sequence_multistart_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load per-sequence candidate-mixture multi-start implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RUN_SEQUENCE_MULTISTART = _IMPL.run_sequence_multistart_candidate_mixture_map
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


_IMPL.run_sequence_multistart_candidate_mixture_map = (
    run_sequence_multistart_candidate_mixture_map
)
globals()["run_sequence_multistart_candidate_mixture_map"] = (
    run_sequence_multistart_candidate_mixture_map
)
globals()["_expand_sequence_less_external_initialization"] = (
    _expand_sequence_less_external_initialization
)
globals()["_first_present_column"] = _first_present_column
globals()["_sequence_id_text"] = _sequence_id_text
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
