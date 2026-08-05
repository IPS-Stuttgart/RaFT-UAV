"""Compatibility package aligning sequence-alignment CLI and API behavior.

The maintained implementation lives in the sibling
``sequence_alignment_audit.py`` module. This package preserves the public import
path while making an omitted ``--sequence-glob`` audit all discovered sequences,
matching :func:`build_sequence_alignment_audit`, and keeping pure-translation
diagnostics separate from axis, sign, and scale alternatives.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "sequence_alignment_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._sequence_alignment_audit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load sequence alignment audit from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_MAIN = _IMPL.main


def _has_sequence_glob(arguments: Sequence[str]) -> bool:
    """Return whether the CLI arguments explicitly select a sequence glob."""

    return any(
        argument == "--sequence-glob" or argument.startswith("--sequence-glob=")
        for argument in arguments
    )


def main(argv: list[str] | None = None) -> int:
    """Run the audit over every sequence unless a glob is explicitly supplied."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not _has_sequence_glob(arguments):
        arguments.extend(["--sequence-glob", "*"])
    return _ORIGINAL_MAIN(arguments)


def _pick_best_translation_row(group: pd.DataFrame) -> pd.Series | None:
    """Return only the as-is median-translation diagnostic row.

    Axis, sign, and scale variants with an additional median translation do not
    isolate a translation error. Letting the best of those rows drive the
    translation diagnosis can misclassify an axis or scale defect as a simple
    offset. If the pure-translation row is absent, return ``None`` so the legacy
    summary falls back to the as-is row instead of making an unsupported claim.
    """

    rows = group.loc[
        group["variant"].astype(str) == "as-is+median-translation"
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


_IMPL.main = main
_IMPL._pick_best_translation_row = _pick_best_translation_row

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_MAIN"] = _ORIGINAL_MAIN
globals()["_has_sequence_glob"] = _has_sequence_glob
globals()["_pick_best_translation_row"] = _pick_best_translation_row
globals()["main"] = main

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
