"""Compatibility package with collision-safe NIS gate probabilities.

The maintained implementation lives in the sibling ``nis_reliability.py``
module. This package preserves the public import path while validating gate
probabilities before early returns and preventing rounded output-column names
from silently overwriting one another.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "nis_reliability.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._nis_reliability_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load NIS reliability implementation from {_LEGACY_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NIS_STATS = _IMPL._nis_stats
_ORIGINAL_NIS_RELIABILITY_SUMMARY = _IMPL.nis_reliability_summary
_ORIGINAL_RUN_NIS_RELIABILITY_REPORT = _IMPL.run_nis_reliability_report


def _normalize_gate_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    """Validate probabilities and reject ambiguous output-column suffixes."""

    if isinstance(values, (str, bytes)):
        raise ValueError("gate_probabilities must be a sequence of numbers in (0, 1)")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(
            "gate_probabilities must be a sequence of numbers in (0, 1)"
        ) from exc

    normalized: list[float] = []
    probability_by_suffix: dict[str, float] = {}
    for index, value in enumerate(items):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "gate_probabilities must contain numbers in (0, 1); "
                f"invalid value at index {index}: {value!r}"
            )
        try:
            probability = _IMPL._validate_probability(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "gate_probabilities must contain numbers in (0, 1); "
                f"invalid value at index {index}: {value!r}"
            ) from exc

        suffix = _IMPL._probability_suffix(probability)
        if suffix in probability_by_suffix:
            previous = probability_by_suffix[suffix]
            if probability != previous:
                raise ValueError(
                    "gate probabilities "
                    f"{previous!r} and {probability!r} map to the same output "
                    f"column suffix {suffix!r}; choose probabilities that remain "
                    "distinct when rounded to three decimal places"
                )
            continue
        probability_by_suffix[suffix] = probability
        normalized.append(probability)
    return tuple(normalized)


def _nis_stats(
    values: np.ndarray,
    *,
    dim: int | None,
    gate_probabilities: Sequence[float],
) -> dict[str, Any]:
    """Compute NIS statistics after validating every requested gate."""

    normalized = _normalize_gate_probabilities(gate_probabilities)
    return _ORIGINAL_NIS_STATS(
        values,
        dim=dim,
        gate_probabilities=normalized,
    )


def nis_reliability_summary(
    frame,
    *,
    group_columns: Sequence[str] = _IMPL.DEFAULT_GROUP_COLUMNS,
    gate_probabilities: Sequence[float] = _IMPL.DEFAULT_GATE_PROBABILITIES,
    accepted_only: bool = False,
):
    """Return NIS reliability statistics with unambiguous gate columns."""

    normalized = _normalize_gate_probabilities(gate_probabilities)
    return _ORIGINAL_NIS_RELIABILITY_SUMMARY(
        frame,
        group_columns=group_columns,
        gate_probabilities=normalized,
        accepted_only=accepted_only,
    )


def run_nis_reliability_report(
    *,
    inputs: Iterable[Path | str],
    output_dir: Path = Path("outputs/nis-reliability"),
    output_name: str = "nis_reliability",
    group_columns: Sequence[str] = _IMPL.DEFAULT_GROUP_COLUMNS,
    gate_probabilities: Sequence[float] = _IMPL.DEFAULT_GATE_PROBABILITIES,
    accepted_only: bool = False,
) -> dict[str, Any]:
    """Write a report whose gate metadata matches its generated columns."""

    normalized = _normalize_gate_probabilities(gate_probabilities)
    return _ORIGINAL_RUN_NIS_RELIABILITY_REPORT(
        inputs=inputs,
        output_dir=output_dir,
        output_name=output_name,
        group_columns=group_columns,
        gate_probabilities=normalized,
        accepted_only=accepted_only,
    )


_IMPL._normalize_gate_probabilities = _normalize_gate_probabilities
_IMPL._nis_stats = _nis_stats
_IMPL.nis_reliability_summary = nis_reliability_summary
_IMPL.run_nis_reliability_report = run_nis_reliability_report

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_gate_probabilities"] = _normalize_gate_probabilities
globals()["_nis_stats"] = _nis_stats
globals()["nis_reliability_summary"] = nis_reliability_summary
globals()["run_nis_reliability_report"] = run_nis_reliability_report

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
