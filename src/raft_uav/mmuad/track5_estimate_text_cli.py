"""Console wrapper preserving textual Track 5 sequence identifiers."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from typing import Any

_impl = importlib.import_module("raft_uav.mmuad.track5_estimate_sequence_" + "gate_fit")
_ORIGINAL_READ_CSV = _impl.pd.read_csv
_SEQUENCE_COLUMNS = ("sequence_id", "Sequence", "sequence", "seq")


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype_arg = kwargs.pop("dtype", None)
    converters = dict(kwargs.pop("converters", {}) or {})
    if dtype_arg is None:
        dtype = {column: "string" for column in _SEQUENCE_COLUMNS}
        _drop_sequence_converters(converters)
    elif isinstance(dtype_arg, Mapping):
        dtype = dict(dtype_arg)
        for column in _SEQUENCE_COLUMNS:
            dtype[column] = "string"
        _drop_sequence_converters(converters)
    else:
        dtype = dtype_arg
        for column in _SEQUENCE_COLUMNS:
            converters[column] = _sequence_id_text
    kwargs["dtype"] = dtype
    if converters:
        kwargs["converters"] = converters
    kwargs.setdefault("keep_default_na", False)
    rows = _ORIGINAL_READ_CSV(path, *args, **kwargs)
    out = rows.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out


def _drop_sequence_converters(converters: dict[Any, Any]) -> None:
    for column in _SEQUENCE_COLUMNS:
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
