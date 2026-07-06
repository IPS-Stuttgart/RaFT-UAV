"""Console wrapper preserving textual sequence IDs for Track 5 gate fitting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from raft_uav.mmuad import track5_sequence_gate_fit as _impl

_ORIGINAL_READ_CSV = _impl.pd.read_csv


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype_arg = kwargs.pop("dtype", None)
    if dtype_arg is None:
        dtype = {"sequence_id": "string"}
    elif isinstance(dtype_arg, Mapping):
        dtype = dict(dtype_arg)
        dtype.setdefault("sequence_id", "string")
    else:
        # pandas accepts scalar dtype arguments such as ``str``.  Do not try to
        # coerce those into a dict; instead keep the caller's scalar dtype and
        # override only sequence_id through a converter so numeric-looking IDs
        # such as ``001`` stay textual.
        dtype = dtype_arg
        converters = dict(kwargs.pop("converters", {}) or {})
        converters.setdefault("sequence_id", _sequence_id_text)
        kwargs["converters"] = converters
    kwargs["dtype"] = dtype
    return _ORIGINAL_READ_CSV(path, *args, **kwargs)


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
