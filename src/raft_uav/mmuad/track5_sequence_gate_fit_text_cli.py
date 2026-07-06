"""Console wrapper preserving textual sequence IDs for Track 5 gate fitting."""

from __future__ import annotations

from typing import Any

from raft_uav.mmuad import track5_sequence_gate_fit as _impl

_ORIGINAL_READ_CSV = _impl.pd.read_csv


def _read_csv_preserving_sequence_id(path: Any, *args: Any, **kwargs: Any):
    dtype = dict(kwargs.pop("dtype", {}) or {})
    dtype.setdefault("sequence_id", "string")
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
