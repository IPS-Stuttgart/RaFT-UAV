"""Text-preserving console wrapper for Track 5 estimate ensembling."""

from __future__ import annotations

from contextvars import ContextVar
from threading import RLock
from typing import Any

from raft_uav.mmuad import track5_estimate_ensemble as _impl

_ORIGINAL_READ_CSV = _impl.pd.read_csv
_PRESERVE_TEXT_CELLS: ContextVar[bool] = ContextVar(
    "raft_uav_preserve_track5_estimate_text_cells",
    default=False,
)
_READ_CSV_PATCH_LOCK = RLock()


def _read_csv_preserving_text_cells(source: Any, *args: Any, **kwargs: Any):
    """Read CSV inputs without pandas coercing opaque official identifiers."""

    kwargs.setdefault("dtype", str)
    kwargs.setdefault("keep_default_na", False)
    return _ORIGINAL_READ_CSV(source, *args, **kwargs)


def _contextual_read_csv(source: Any, *args: Any, **kwargs: Any):
    """Preserve text only for the active ensemble invocation context."""

    if _PRESERVE_TEXT_CELLS.get():
        return _read_csv_preserving_text_cells(source, *args, **kwargs)
    return _ORIGINAL_READ_CSV(source, *args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    """Run the text-safe CLI without changing CSV semantics in other threads."""

    with _READ_CSV_PATCH_LOCK:
        original = _impl.pd.read_csv
        token = _PRESERVE_TEXT_CELLS.set(True)
        _impl.pd.read_csv = _contextual_read_csv
        try:
            return _impl.main(argv)
        finally:
            _impl.pd.read_csv = original
            _PRESERVE_TEXT_CELLS.reset(token)


__all__ = ["main", "_read_csv_preserving_text_cells"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
