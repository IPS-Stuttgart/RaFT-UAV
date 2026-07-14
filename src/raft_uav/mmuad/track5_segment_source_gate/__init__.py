"""Package wrapper that hardens Track 5 segment source-gate inputs.

The implementation lives in the sibling ``track5_segment_source_gate.py`` file.
This wrapper keeps the public import path while accepting spreadsheet-exported
template DataFrames with whitespace around alias headers, canonicalizing opaque
sequence IDs with the shared official Track 5 parser, and rejecting malformed
source-gate controls before they can disable penalties or corrupt path costs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.mmuad.submission import parse_official_sequence_cell

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_segment_source_gate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_segment_source_gate_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 segment source-gate implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_TEMPLATE_ROWS_ATTR = "_raft_uav_original_normalize_template_rows"
_ORIGINAL_FIRST_PRESENT_ATTR = "_raft_uav_original_first_present"
_ORIGINAL_BUILD_ATTR = "_raft_uav_original_build_track5_segment_source_gate"
_ORIGINAL_WRITE_ATTR = "_raft_uav_original_write_track5_segment_source_gate_outputs"

if not hasattr(_IMPL, _ORIGINAL_NORMALIZE_TEMPLATE_ROWS_ATTR):
    setattr(
        _IMPL,
        _ORIGINAL_NORMALIZE_TEMPLATE_ROWS_ATTR,
        _IMPL._normalize_template_rows,
    )
if not hasattr(_IMPL, _ORIGINAL_FIRST_PRESENT_ATTR):
    setattr(_IMPL, _ORIGINAL_FIRST_PRESENT_ATTR, _IMPL._first_present)
if not hasattr(_IMPL, _ORIGINAL_BUILD_ATTR):
    setattr(_IMPL, _ORIGINAL_BUILD_ATTR, _IMPL.build_track5_segment_source_gate)
if not hasattr(_IMPL, _ORIGINAL_WRITE_ATTR):
    setattr(
        _IMPL,
        _ORIGINAL_WRITE_ATTR,
        _IMPL.write_track5_segment_source_gate_outputs,
    )


_ORIGINAL_BUILD = getattr(_IMPL, _ORIGINAL_BUILD_ATTR)
_ORIGINAL_WRITE = getattr(_IMPL, _ORIGINAL_WRITE_ATTR)


def _official_sequence_text(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _first_present_with_stripped_headers(rows: Any, names: tuple[str, ...]) -> Any | None:
    lower = {str(column).strip().casefold(): column for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(str(name).casefold())
        if found is not None:
            return found
    return None


def _normalize_template_rows_with_official_sequence_ids(template: Any) -> Any:
    rows = _IMPL.pd.DataFrame(template).copy()
    sequence_column = _first_present_with_stripped_headers(
        rows,
        ("sequence_id", "Sequence", "sequence", "seq"),
    )
    time_column = _first_present_with_stripped_headers(
        rows,
        ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"),
    )
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = _IMPL.pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].map(_official_sequence_text),
            "time_s": _IMPL.pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & _IMPL.np.isfinite(out["time_s"].to_numpy(float))
    return (
        out.loc[finite]
        .drop_duplicates()
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def _finite_config_value(value: Any, *, name: str) -> float:
    """Return one finite non-Boolean scalar for a source-gate control."""

    message = f"{name} must be a finite scalar"
    if isinstance(value, _IMPL.np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    elif isinstance(value, _IMPL.np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not _IMPL.np.isfinite(number):
        raise ValueError(message)
    return number


def _validated_segment_source_gate_config(config: Any | None) -> Any:
    """Validate controls before any source resampling or dynamic programming."""

    resolved = config or _IMPL.SegmentSourceGateConfig()
    for name in ("speed_limit_mps", "acceleration_limit_mps2", "invalid_penalty"):
        value = _finite_config_value(getattr(resolved, name), name=name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    for name in (
        "switch_penalty",
        "switch_jump_penalty_per_m",
        "weight_log_scale",
    ):
        value = _finite_config_value(getattr(resolved, name), name=name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")
    return resolved


def build_track5_segment_source_gate(
    estimate_inputs: Any,
    template: Any,
    *,
    config: Any | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[Any, Any]:
    """Build a source-gated estimate after validating every cost control."""

    return _ORIGINAL_BUILD(
        estimate_inputs,
        template,
        config=_validated_segment_source_gate_config(config),
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


def write_track5_segment_source_gate_outputs(
    *,
    estimate_inputs: Any,
    template: Any,
    output_dir: Any,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    config: Any | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Any]:
    """Write source-gate artifacts only from a valid cost configuration."""

    return _ORIGINAL_WRITE(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=output_dir,
        class_map=class_map,
        default_classification=default_classification,
        config=_validated_segment_source_gate_config(config),
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


_IMPL._first_present = _first_present_with_stripped_headers
_IMPL._normalize_template_rows = _normalize_template_rows_with_official_sequence_ids
_IMPL._finite_config_value = _finite_config_value
_IMPL._validated_segment_source_gate_config = _validated_segment_source_gate_config
_IMPL.build_track5_segment_source_gate = build_track5_segment_source_gate
_IMPL.write_track5_segment_source_gate_outputs = write_track5_segment_source_gate_outputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

_first_present = _first_present_with_stripped_headers
_normalize_template_rows = _normalize_template_rows_with_official_sequence_ids
_finite_config_value = _finite_config_value
_validated_segment_source_gate_config = _validated_segment_source_gate_config
build_track5_segment_source_gate = build_track5_segment_source_gate
write_track5_segment_source_gate_outputs = write_track5_segment_source_gate_outputs
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
