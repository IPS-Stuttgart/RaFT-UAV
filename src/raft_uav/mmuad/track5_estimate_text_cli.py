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
_SEQUENCE_COLUMN_KEYS = {column.casefold() for column in _SEQUENCE_COLUMNS}


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype_arg = kwargs.pop("dtype", None)
    converters = dict(kwargs.pop("converters", {}) or {})
    sequence_columns = _sequence_columns_for_read(path, args, kwargs)
    if dtype_arg is None:
        dtype = {column: "string" for column in sequence_columns}
        _drop_sequence_converters(converters, sequence_columns)
    elif isinstance(dtype_arg, Mapping):
        dtype = dict(dtype_arg)
        for column in sequence_columns:
            dtype[column] = "string"
        _drop_sequence_converters(converters, sequence_columns)
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
    out.columns = [str(column).strip() for column in out.columns]
    return out


def _sequence_columns_for_read(path: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    discovered = _discover_sequence_columns(path, args, kwargs)
    return tuple(dict.fromkeys((*_SEQUENCE_COLUMNS, *discovered)))


def _discover_sequence_columns(path: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    explicit_names = kwargs.get("names")
    if explicit_names is not None:
        return tuple(column for column in explicit_names if _is_sequence_column(column))

    peek_kwargs = dict(kwargs)
    peek_kwargs.pop("dtype", None)
    peek_kwargs.pop("converters", None)
    peek_kwargs["nrows"] = 0
    peek_kwargs.setdefault("keep_default_na", False)

    position = None
    if hasattr(path, "tell") and hasattr(path, "seek"):
        try:
            position = path.tell()
        except OSError:
            position = None
    try:
        header = _ORIGINAL_READ_CSV(path, *args, **peek_kwargs)
    except Exception:
        return ()
    finally:
        if position is not None:
            try:
                path.seek(position)
            except OSError:
                pass
    return tuple(column for column in header.columns if _is_sequence_column(column))


def _is_sequence_column(column: Any) -> bool:
    return str(column).strip().casefold() in _SEQUENCE_COLUMN_KEYS


def _drop_sequence_converters(converters: dict[Any, Any], sequence_columns: tuple[Any, ...]) -> None:
    for column in sequence_columns:
        converters.pop(column, None)


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
