"""Compatibility package with validated reservoir offset-grid specifications.

The maintained implementation lives in the sibling
``candidate_reservoir_grid.py`` module. This package preserves the public import
path while rejecting ambiguous or non-finite offset-grid specifications, removing
repeated values that would otherwise rerun identical configurations, keeping
distinct floating-point offsets distinct in per-configuration labels, validating
programmatic grid controls without lossy coercion, and failing closed when a
truth-backed grid requests a best reservoir without a finite selection score.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float as _shared_optional_float
from raft_uav.numeric import optional_int as _shared_optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_grid.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_grid_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate reservoir grid from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_LEGACY_OFFSET_CONFIG_GRID = _IMPL._offset_config_grid


def _parse_offset_specs(specs: Sequence[str]) -> list[tuple[str, tuple[float, ...]]]:
    """Parse unique finite offset grids without redundant configurations."""

    parsed: list[tuple[str, tuple[float, ...]]] = []
    seen_names: set[str] = set()
    seen_label_names: dict[str, str] = {}
    for spec in specs:
        text = str(spec)
        if "=" not in text:
            raise ValueError(f"offset grid spec must be NAME=v1,v2,...; got {spec!r}")
        name, values_text = text.split("=", 1)
        name = name.strip()
        value_tokens = [token.strip() for token in values_text.split(",")]
        if not name or not value_tokens or any(not token for token in value_tokens):
            raise ValueError(f"invalid offset grid spec {spec!r}")
        if name == "__none__":
            raise ValueError("offset grid name '__none__' is reserved for the identity grid")
        if name in seen_names:
            raise ValueError(f"duplicate offset grid name {name!r}")
        label_name = _IMPL._sanitize_label(name)
        previous_name = seen_label_names.get(label_name)
        if previous_name is not None:
            raise ValueError(
                "offset grid names must remain unique after filename normalization: "
                f"{previous_name!r} and {name!r}"
            )

        values: list[float] = []
        for token in value_tokens:
            try:
                value = float(token)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid offset value {token!r} for {name!r}") from exc
            if not np.isfinite(value):
                raise ValueError(f"offset values for {name!r} must be finite")
            if value not in values:
                values.append(value)

        seen_names.add(name)
        seen_label_names[label_name] = name
        parsed.append((name, tuple(values)))
    return parsed


def _offset_config_grid(
    branch_specs: Sequence[str],
    source_specs: Sequence[str],
) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    """Build offset configurations while enforcing unique serialized labels."""

    configs = _LEGACY_OFFSET_CONFIG_GRID(branch_specs, source_specs)
    seen: dict[
        str,
        tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]],
    ] = {}
    for label, branch_offsets, source_offsets in configs:
        identity = (
            tuple(sorted(branch_offsets.items())),
            tuple(sorted(source_offsets.items())),
        )
        previous = seen.get(label)
        if previous is not None and previous != identity:
            raise ValueError(
                f"offset grid configurations collide after label formatting: {label!r}"
            )
        seen[label] = identity
    return configs


def _format_float(value: float) -> str:
    """Return a filename-safe shortest round-trip floating-point label."""

    text = repr(float(value))
    if text.endswith(".0") and "e" not in text.lower():
        text = text[:-2]
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _optional_int_control(value: object) -> int | None:
    """Parse an integer-like scalar while preserving the helper's failure contract."""

    try:
        return _shared_optional_int(value)
    except (OverflowError, TypeError, ValueError):
        return None


def _nonnegative_integer_control(name: str, value: object) -> int:
    number = _optional_int_control(value)
    if number is None or number < 0:
        raise ValueError(f"{name} must be a non-negative integer scalar")
    return number


def _finite_float_control(name: str, value: object) -> float:
    number = _shared_optional_float(value)
    if number is None:
        raise ValueError(f"{name} must be a finite real scalar")
    return number


def _nonnegative_float_control(name: str, value: object) -> float:
    number = _finite_float_control(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return number


def _score_floor_quantile_control(value: object | None) -> float | None:
    if value is None:
        return None
    number = _finite_float_control("score_floor_quantile", value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("score_floor_quantile must be a finite real scalar in [0, 1]")
    return number


def _top_k_control(values: Sequence[object]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("top_k_values must be a non-empty sequence of positive integers")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "top_k_values must be a non-empty sequence of positive integers"
        ) from exc
    if not raw_values:
        raise ValueError("top_k_values must be a non-empty sequence of positive integers")

    parsed: list[int] = []
    for value in raw_values:
        number = _optional_int_control(value)
        if number is None or number <= 0:
            raise ValueError("top_k_values must contain only positive integer scalars")
        parsed.append(number)
    return tuple(sorted(set(parsed)))


def _boolean_control(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a Boolean scalar")
    return bool(value)


def _selection_metric_control(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("selection_metric must be a non-empty string")
    return value.strip()


def _require_finite_selection_metric(summary: pd.DataFrame, selection_metric: str) -> None:
    if selection_metric not in summary.columns:
        raise ValueError(
            f"selection metric {selection_metric!r} was not produced; "
            "check top_k_values and truth/candidate overlap"
        )
    values = pd.to_numeric(summary[selection_metric], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(
            f"selection metric {selection_metric!r} has no finite values; "
            "no reservoir grid candidate was scored against the supplied truth"
        )


def run_candidate_reservoir_offset_grid(
    candidates: pd.DataFrame,
    *,
    truth: pd.DataFrame | None = None,
    branch_offset_grid: Sequence[str] = (),
    source_offset_grid: Sequence[str] = (),
    output_dir: Path | None = None,
    score_column: str = "ranker_score",
    fallback_score_column: str = "confidence",
    global_top_n: object = 20,
    per_source_top_n: object = 3,
    per_branch_top_n: object = 3,
    max_candidates_per_frame: object = 40,
    score_floor_quantile: object | None = None,
    cap_reason_bonus: object = 0.0,
    top_k_values: Sequence[object] = _IMPL._DEFAULT_TOP_K,
    max_truth_time_delta_s: object = 0.5,
    selection_metric: object = _IMPL._DEFAULT_SELECTION_METRIC,
    write_best_reservoir: object = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run a reservoir offset grid after strict control validation."""

    validated_global_top_n = _nonnegative_integer_control("global_top_n", global_top_n)
    validated_per_source_top_n = _nonnegative_integer_control(
        "per_source_top_n",
        per_source_top_n,
    )
    validated_per_branch_top_n = _nonnegative_integer_control(
        "per_branch_top_n",
        per_branch_top_n,
    )
    validated_max_candidates = _nonnegative_integer_control(
        "max_candidates_per_frame",
        max_candidates_per_frame,
    )
    validated_score_floor = _score_floor_quantile_control(score_floor_quantile)
    validated_cap_reason_bonus = _finite_float_control("cap_reason_bonus", cap_reason_bonus)
    validated_top_k = _top_k_control(top_k_values)
    validated_truth_delta = _nonnegative_float_control(
        "max_truth_time_delta_s",
        max_truth_time_delta_s,
    )
    validated_selection_metric = _selection_metric_control(selection_metric)
    validated_write_best = _boolean_control("write_best_reservoir", write_best_reservoir)

    rows = pd.DataFrame(candidates).copy()
    if rows.empty:
        raise ValueError("candidate reservoir offset grid requires candidate rows")
    truth_rows = (
        None
        if truth is None
        else _IMPL.normalize_truth_columns(pd.DataFrame(truth).copy())
    )
    offset_configs = _offset_config_grid(branch_offset_grid, source_offset_grid)
    summary_records: list[dict[str, Any]] = []
    reservoirs: dict[str, pd.DataFrame] = {}
    frame_tables: dict[str, pd.DataFrame] = {}
    by_sequence_tables: dict[str, pd.DataFrame] = {}

    for index, (label, branch_offsets, source_offsets) in enumerate(offset_configs, start=1):
        adjusted = _IMPL._with_adjusted_scores(
            rows,
            branch_offsets=branch_offsets,
            source_offsets=source_offsets,
            score_column=score_column,
            fallback_score_column=fallback_score_column,
        )
        reservoir = _IMPL.build_candidate_reservoir(
            adjusted,
            config=_IMPL.ReservoirConfig(
                global_top_n=validated_global_top_n,
                per_source_top_n=validated_per_source_top_n,
                per_branch_top_n=validated_per_branch_top_n,
                max_candidates_per_frame=validated_max_candidates,
                score_column="candidate_reservoir_grid_score",
                fallback_score_column=fallback_score_column,
                score_floor_quantile=validated_score_floor,
                cap_reason_bonus=validated_cap_reason_bonus,
            ),
        )
        summary: dict[str, Any] = {
            "grid_index": int(index),
            "grid_label": label,
            "branch_score_offsets_json": _IMPL.json.dumps(branch_offsets, sort_keys=True),
            "source_score_offsets_json": _IMPL.json.dumps(source_offsets, sort_keys=True),
            "cap_reason_bonus": validated_cap_reason_bonus,
        }
        summary |= _IMPL.build_reservoir_summary(rows, reservoir)
        if truth_rows is not None:
            frame_rows, pooled, by_sequence = _IMPL.build_oracle_recall_tables(
                reservoir,
                truth_rows,
                top_k_values=validated_top_k,
                max_truth_time_delta_s=validated_truth_delta,
            )
            if not pooled.empty:
                summary |= pooled.iloc[0].to_dict()
            frame_tables[label] = frame_rows
            by_sequence_tables[label] = by_sequence
        summary_records.append(summary)
        reservoirs[label] = reservoir

    summary_frame = pd.DataFrame.from_records(summary_records)
    if truth_rows is not None and validated_write_best:
        _require_finite_selection_metric(summary_frame, validated_selection_metric)
    summary_frame = _IMPL._sort_summary(
        summary_frame,
        selection_metric=validated_selection_metric,
    )

    best_reservoir: pd.DataFrame | None = None
    if not summary_frame.empty:
        best_label = str(summary_frame.iloc[0]["grid_label"])
        best_reservoir = reservoirs.get(best_label)

    if output_dir is not None:
        _IMPL._write_outputs(
            output_dir=Path(output_dir),
            summary_frame=summary_frame,
            reservoirs=reservoirs,
            frame_tables=frame_tables,
            by_sequence_tables=by_sequence_tables,
            best_reservoir=best_reservoir if validated_write_best else None,
        )
    return summary_frame, best_reservoir if validated_write_best else None


_IMPL._parse_offset_specs = _parse_offset_specs
_IMPL._offset_config_grid = _offset_config_grid
_IMPL._format_float = _format_float
_IMPL.run_candidate_reservoir_offset_grid = run_candidate_reservoir_offset_grid

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_parse_offset_specs"] = _parse_offset_specs
globals()["_offset_config_grid"] = _offset_config_grid
globals()["_format_float"] = _format_float
globals()["_optional_int_control"] = _optional_int_control
globals()["_nonnegative_integer_control"] = _nonnegative_integer_control
globals()["_finite_float_control"] = _finite_float_control
globals()["_nonnegative_float_control"] = _nonnegative_float_control
globals()["_score_floor_quantile_control"] = _score_floor_quantile_control
globals()["_top_k_control"] = _top_k_control
globals()["_boolean_control"] = _boolean_control
globals()["_selection_metric_control"] = _selection_metric_control
globals()["_require_finite_selection_metric"] = _require_finite_selection_metric
globals()["run_candidate_reservoir_offset_grid"] = run_candidate_reservoir_offset_grid

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
