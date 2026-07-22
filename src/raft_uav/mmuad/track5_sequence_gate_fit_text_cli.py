"""Console wrapper preserving textual sequence IDs for Track 5 gate fitting."""

from __future__ import annotations

from collections.abc import Mapping
import threading
from typing import Any

from raft_uav.mmuad import track5_sequence_gate_fit as _impl

_ORIGINAL_PANDAS = _impl.pd
_ORIGINAL_READ_CSV = _ORIGINAL_PANDAS.read_csv
_MAIN_LOCK = threading.RLock()
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


class _SequencePreservingPandasProxy:
    """Delegate pandas operations while overriding only this implementation's CSV reads."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_ORIGINAL_PANDAS, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any):
        return _read_csv_preserving_sequence_id(path, *args, **kwargs)


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype_arg = kwargs.pop("dtype", None)
    converters = dict(kwargs.pop("converters", {}) or {})
    sequence_columns = _sequence_columns_for_csv(path, *args, **kwargs)
    _drop_sequence_converters(converters)
    if dtype_arg is None:
        dtype = "string"
    elif isinstance(dtype_arg, Mapping):
        dtype = {
            column: value
            for column, value in dtype_arg.items()
            if not _is_sequence_column(column)
        }
        for column in sequence_columns:
            converters[column] = _sequence_id_text
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


def _sequence_columns_for_csv(path: Any, *args: Any, **kwargs: Any) -> list[str]:
    discovered = _discover_sequence_columns_for_csv(path, *args, **kwargs)
    return list(dict.fromkeys([*_SEQUENCE_ID_ALIASES, *discovered]))


def _discover_sequence_columns_for_csv(path: Any, *args: Any, **kwargs: Any) -> list[str]:
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
        str(column)
        for column in header.columns
        if _is_sequence_column(column)
    ]


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


def _drop_sequence_converters(converters: dict[Any, Any]) -> None:
    for column in list(converters):
        if _is_sequence_column(column):
            converters.pop(column, None)


def _is_sequence_column(column: Any) -> bool:
    return str(column).strip().lower() in _SEQUENCE_ID_ALIAS_KEYS


def _sequence_id_text(value: Any) -> str:
    return "" if value is None else str(value)


def main(argv: list[str] | None = None) -> int:
    with _MAIN_LOCK:
        original_impl_pd = _impl.pd
        _impl.pd = _SequencePreservingPandasProxy()
        try:
            return _impl.main(argv)
        finally:
            _impl.pd = original_impl_pd


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
