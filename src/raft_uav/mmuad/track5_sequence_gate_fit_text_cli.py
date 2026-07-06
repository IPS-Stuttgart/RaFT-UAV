"""Console wrapper preserving textual sequence IDs for Track 5 gate fitting."""

from __future__ import annotations

from collections.abc import Mapping
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
    "heldout_sequence",
)


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype_arg = kwargs.pop("dtype", None)
    converters = dict(kwargs.pop("converters", {}) or {})
    if dtype_arg is None:
        dtype = {alias: "string" for alias in _SEQUENCE_ID_ALIASES}
        _drop_sequence_converters(converters)
    elif isinstance(dtype_arg, Mapping):
        dtype = dict(dtype_arg)
        for alias in _SEQUENCE_ID_ALIASES:
            dtype[alias] = "string"
        _drop_sequence_converters(converters)
    else:
        dtype = dtype_arg
        for alias in _SEQUENCE_ID_ALIASES:
            converters[alias] = _sequence_id_text
    kwargs["dtype"] = dtype
    if converters:
        kwargs["converters"] = converters
    return _ORIGINAL_READ_CSV(path, *args, **kwargs)


def _drop_sequence_converters(converters: dict[Any, Any]) -> None:
    for alias in _SEQUENCE_ID_ALIASES:
        converters.pop(alias, None)


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
