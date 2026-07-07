"""Text-preserving console wrapper for Track 5 estimate ensembling."""

from __future__ import annotations

from typing import Any

from raft_uav.mmuad import track5_estimate_ensemble as _impl

_ORIGINAL_READ_CSV = _impl.pd.read_csv


def _read_csv_preserving_text_cells(source: Any, *args: Any, **kwargs: Any):
    """Read CSV inputs without pandas coercing opaque official identifiers."""

    kwargs.setdefault("dtype", str)
    kwargs.setdefault("keep_default_na", False)
    return _ORIGINAL_READ_CSV(source, *args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    original = _impl.pd.read_csv
    _impl.pd.read_csv = _read_csv_preserving_text_cells
    try:
        return _impl.main(argv)
    finally:
        _impl.pd.read_csv = original


__all__ = ["main", "_read_csv_preserving_text_cells"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
