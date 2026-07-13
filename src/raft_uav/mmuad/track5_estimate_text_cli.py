"""Console wrapper preserving textual Track 5 sequence identifiers."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from typing import Any

_impl = importlib.import_module("raft_uav.mmuad.track5_estimate_sequence_" + "gate_fit")
_ORIGINAL_READ_CSV = _impl.pd.read_csv
_SEQUENCE_COLUMNS = (
    "sequence_id",
    "Sequence",
    "sequence",
    "seq",
    "scene",
    "scene_id",
    "clip",
    "clip_id",
)
_SEQUENCE_COLUMN_KEYS = frozenset(column.strip().lower() for column in _SEQUENCE_COLUMNS)


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype_arg = kwargs.pop("dtype", None)
    converters = dict(kwargs.pop("converters", {}) or {})
    sequence_columns = _sequence_columns_for_csv(path, *args, **kwargs)
    for column in sequence_columns:
        converters.pop(column, None)
    if dtype_arg is None:
        dtype = {column: "string" for column in sequence_columns}
    elif isinstance(dtype_arg, Mapping):
        dtype = dict(dtype_arg)
        for column in sequence_columns:
            dtype[column] = "string"
    else:
        dtype = dtype_arg
        for column in sequence_columns:
            converters[column] = _sequence_id_text
    kwargs["dtype"] = dtype
    if converters:
        kwargs["converters"] = converters
    kwargs.setdefault("keep_default_na", False)
    rows = _ORIGINAL_READ_CSV(path, *args, **kwargs)
    out = rows.copy()
    out.columns = _normalized_column_names(out.columns)
    return out


def _normalized_column_names(columns: Any) -> list[str]:
    normalized = [str(column).strip() for column in columns]
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in normalized:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    if duplicates:
        joined = ", ".join(repr(column) for column in duplicates)
        raise ValueError(
            "CSV columns are ambiguous after whitespace normalization: " + joined
        )
    return normalized


def _sequence_columns_for_csv(path: Any, *args: Any, **kwargs: Any) -> list[str]:
    discovered = _discover_sequence_columns_for_csv(path, *args, **kwargs)
    return list(dict.fromkeys([*_SEQUENCE_COLUMNS, *discovered]))


def _discover_sequence_columns_for_csv(path: Any, *args: Any, **kwargs: Any) -> list[str]:
    header_kwargs = dict(kwargs)
    header_kwargs.pop("dtype", None)
    header_kwargs.pop("converters", None)
    header_kwargs["nrows"] = 0
    position = _stream_position(path)
    try:
        header = _ORIGINAL_READ_CSV(path, *args, **header_kwargs)
    except Exception:
        return []
    finally:
        if position is not None:
            _restore_stream_position(path, position)
    return [
        str(column)
        for column in header.columns
        if str(column).strip().lower() in _SEQUENCE_COLUMN_KEYS
    ]


def _stream_position(path: Any) -> int | None:
    if not (hasattr(path, "tell") and hasattr(path, "seek")):
        return None
    try:
        return int(path.tell())
    except (OSError, TypeError, ValueError):
        return None


def _restore_stream_position(path: Any, position: int) -> None:
    try:
        path.seek(position)
    except (OSError, TypeError, ValueError):
        pass


def _sequence_id_text(value: Any) -> str:
    return "" if value is None else str(value)


def main(argv: list[str] | None = None) -> int:
    original = _impl.pd.read_csv
    _impl.pd.read_csv = _read_csv_preserving_sequence_id
    try:
        return _impl.main(argv)
    finally:
        _impl.pd.read_csv = original


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
