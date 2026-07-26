"""Compatibility guards for NIS reliability report inputs and output columns.

The maintained implementation lives in the sibling ``nis_reliability.py`` module.
This package preserves the public import path while rejecting measurement dimensions
that are not exact positive integers, preserving opaque CSV identifiers, and rejecting
gate-probability sets that would overwrite one another after column-suffix formatting.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "nis_reliability.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._nis_reliability_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load NIS reliability implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NIS_RELIABILITY_SUMMARY = _IMPL.nis_reliability_summary
_ORIGINAL_NORMALIZED_NIS_FRAME = _IMPL._normalized_nis_frame


def _exact_measurement_dimension_mask(values: pd.Series) -> np.ndarray:
    """Return rows whose dimensions are non-Boolean exact positive integers."""

    raw = pd.Series(values)
    boolean = raw.map(lambda value: isinstance(value, (bool, np.bool_))).to_numpy(dtype=bool)
    numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    return (
        np.isfinite(numeric)
        & (numeric > 0.0)
        & (numeric == np.rint(numeric))
        & ~boolean
    )


def _normalized_nis_frame(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    accepted_only: bool,
) -> pd.DataFrame:
    """Normalize NIS rows without rounding near-integer or Boolean dimensions."""

    prepared = frame
    validity_column: object | None = None
    if "measurement_dim" in frame.columns:
        prepared = frame.copy()
        validity_column = object()
        prepared[validity_column] = _exact_measurement_dimension_mask(
            frame["measurement_dim"]
        )

    normalized = _ORIGINAL_NORMALIZED_NIS_FRAME(
        prepared,
        group_columns=group_columns,
        accepted_only=accepted_only,
    )
    if validity_column is None:
        valid_dimension = _exact_measurement_dimension_mask(normalized["measurement_dim"])
    else:
        valid_dimension = normalized.pop(validity_column).to_numpy(dtype=bool)
    return normalized.loc[valid_dimension].copy()


def read_nis_diagnostics(paths: Iterable[Path | str]) -> pd.DataFrame:
    """Read diagnostics CSVs without coercing opaque identifier columns.

    Per-sequence reliability reports may use numeric-looking identifiers such as
    ``001``. Reading each file as text first prevents pandas from collapsing such
    identifiers before the requested grouping is applied. NIS and dimension columns
    are converted by the normal summary pipeline.
    """

    frames: list[pd.DataFrame] = []
    for path_like in paths:
        path = Path(path_like)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["diagnostics_path"] = str(path)
        if "measurement_dim" not in frame.columns:
            frame["measurement_dim"] = _IMPL._infer_measurement_dim(frame)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _validated_gate_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    """Return valid probabilities whose formatted output suffixes are unique."""

    probabilities = tuple(_IMPL._validate_probability(value) for value in values)
    values_by_suffix: dict[str, list[float]] = {}
    for probability in probabilities:
        suffix = _IMPL._probability_suffix(probability)
        values_by_suffix.setdefault(suffix, []).append(probability)

    collisions = {
        suffix: suffix_values
        for suffix, suffix_values in values_by_suffix.items()
        if len(suffix_values) > 1
    }
    if collisions:
        rendered = "; ".join(
            f"{suffix}: {', '.join(repr(value) for value in suffix_values)}"
            for suffix, suffix_values in sorted(collisions.items())
        )
        raise ValueError(
            "gate probabilities produce duplicate output column suffixes: "
            f"{rendered}"
        )
    return probabilities


def nis_reliability_summary(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = _IMPL.DEFAULT_GROUP_COLUMNS,
    gate_probabilities: Sequence[float] = _IMPL.DEFAULT_GATE_PROBABILITIES,
    accepted_only: bool = False,
) -> pd.DataFrame:
    """Return NIS statistics only when dimensions and gate columns are unambiguous."""

    validated_probabilities = _validated_gate_probabilities(gate_probabilities)
    return _ORIGINAL_NIS_RELIABILITY_SUMMARY(
        frame,
        group_columns=group_columns,
        gate_probabilities=validated_probabilities,
        accepted_only=accepted_only,
    )


_IMPL._normalized_nis_frame = _normalized_nis_frame
_IMPL.read_nis_diagnostics = read_nis_diagnostics
_IMPL.nis_reliability_summary = nis_reliability_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_exact_measurement_dimension_mask"] = _exact_measurement_dimension_mask
globals()["_normalized_nis_frame"] = _normalized_nis_frame
globals()["read_nis_diagnostics"] = read_nis_diagnostics
globals()["_validated_gate_probabilities"] = _validated_gate_probabilities
globals()["nis_reliability_summary"] = nis_reliability_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
