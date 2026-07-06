"""Console wrapper preserving textual Track 5 sequence identifiers."""

from __future__ import annotations

import importlib
from typing import Any

_impl = importlib.import_module("raft_uav.mmuad.track5_estimate_sequence_" + "gate_fit")
_ORIGINAL_READ_CSV = _impl.pd.read_csv
_SEQUENCE_COLUMNS = ("sequence_id", "Sequence", "sequence", "seq")


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype = dict(kwargs.pop("dtype", {}) or {})
    for column in _SEQUENCE_COLUMNS:
        dtype.setdefault(column, "string")
    kwargs["dtype"] = dtype
    return _ORIGINAL_READ_CSV(path, *args, **kwargs)


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
