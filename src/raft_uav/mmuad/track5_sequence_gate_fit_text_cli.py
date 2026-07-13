"""Console wrapper preserving textual sequence IDs for Track 5 gate fitting."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

from raft_uav.mmuad import track5_sequence_gate_fit as _impl

_ORIGINAL_READ_CSV = _impl.pd.read_csv
_SEQUENCE_ID_ALIASES = (
    "sequence_id",
    "Sequence",
    "sequence",
    "seq",
    "scene",
    "scene_id",
    "clip",
    "clip_id",
    "heldout_sequence",
)
_SEQUENCE_ID_ALIAS_KEYS = frozenset(alias.strip().lower() for alias in _SEQUENCE_ID_ALIASES)


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype_arg = kwargs.pop("dtype", None)
    converters = dict(kwargs.pop("converters", {}) or {})
    sequence_columns = _sequence_columns_for_csv(path, *args, **kwargs)
    sequence_names = frozenset(column for _, column in sequence_columns)
    sequence_positions = frozenset(position for position, _ in sequence_columns)
    _drop_sequence_converters(
        converters,
        sequence_names=sequence_names,
        sequence_positions=sequence_positions,
    )
    if dtype_arg is None:
        dtype = "string"
    elif isinstance(dtype_arg, Mapping):
        dtype = dict(dtype_arg)
        for column in list(dtype):
            if _is_sequence_key(
                column,
                sequence_names=sequence_names,
                sequence_positions=sequence_positions,
            ):
                dtype[column] = "string"
        for alias in _SEQUENCE_ID_ALIASES:
            dtype[alias] = "string"
        for column in sequence_names:
            dtype[column] = "string"
    else:
        dtype = dtype_arg
        for alias in _SEQUENCE_ID_ALIASES:
            converters[alias] = _sequence_id_text
        for column in sequence_names:
            converters[column] = _sequence_id_text
    kwargs["dtype"] = dtype
    if converters:
        kwargs["converters"] = converters
    kwargs.setdefault("keep_default_na", False)
    rows = _ORIGINAL_READ_CSV(path, *args, **kwargs)
    out = rows.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out


def _sequence_columns_for_csv(path: Any, *args: Any, **kwargs: Any) -> list[tuple[int, str]]:
    position = _stream_position(path)
    if hasattr(path, "read") and position is None:
        return []
    header_kwargs = dict(kwargs)
    header_kwargs.pop("dtype", None)
    header_kwargs.pop("converters", None)
    header_kwargs.pop("chunksize", None)
    header_kwargs.pop("iterator", None)
    header_kwargs["nrows"] = 0
    try:
        header = _ORIGINAL_READ_CSV(path, *args, **header_kwargs)
    except Exception:
        return []
    finally:
        if position is not None:
            _restore_stream_position(path, position)
    return [
        (position, str(column))
        for position, column in enumerate(header.columns)
        if _is_sequence_column(column)
    ]


def _drop_sequence_converters(
    converters: dict[Any, Any],
    *,
    sequence_names: frozenset[str] = frozenset(),
    sequence_positions: frozenset[int] = frozenset(),
) -> None:
    for column in list(converters):
        if _is_sequence_key(
            column,
            sequence_names=sequence_names,
            sequence_positions=sequence_positions,
        ):
            converters.pop(column, None)


def _is_sequence_key(
    column: Any,
    *,
    sequence_names: frozenset[str],
    sequence_positions: frozenset[int],
) -> bool:
    if _is_sequence_column(column) or column in sequence_names:
        return True
    return isinstance(column, Integral) and int(column) in sequence_positions


def _is_sequence_column(column: Any) -> bool:
    return str(column).strip().lower() in _SEQUENCE_ID_ALIAS_KEYS


def _stream_position(path: Any) -> int | None:
    if not (hasattr(path, "tell") and hasattr(path, "seek")):
        return None
    try:
        position = int(path.tell())
        path.seek(position)
    except (OSError, TypeError, ValueError):
        return None
    return position


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
