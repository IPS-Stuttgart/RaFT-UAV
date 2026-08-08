"""Compatibility package guarding tournament selected-prediction copies.

The maintained implementation lives in the sibling ``tournament.py`` module.
This package preserves the public import path while preventing the destructive
``selected_predictions`` refresh from deleting its own selected candidate input.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tournament.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.multi_uav_lts._tournament_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load tournament implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_COPY_SELECTED_PREDICTIONS = _IMPL._copy_selected_predictions


def _selected_prediction_copy_target(path: Path, output_dir: Path) -> Path:
    """Return the path replaced by the tournament's selected-output refresh."""

    if path.is_dir():
        return output_dir / "selected_predictions"
    suffix = "".join(path.suffixes) or ".bin"
    return output_dir / f"selected_predictions{suffix}"


def _copy_selected_predictions(path: Path, output_dir: Path) -> None:
    """Copy a selected candidate without deleting that candidate first."""

    source = Path(path)
    destination_root = Path(output_dir)
    target = _selected_prediction_copy_target(source, destination_root)
    source_resolved = source.resolve()
    target_resolved = target.resolve()
    if source_resolved == target_resolved or source_resolved.is_relative_to(
        target_resolved
    ):
        raise ValueError(
            "selected candidate path must not be the selected_predictions copy "
            "target or live inside it"
        )
    _ORIGINAL_COPY_SELECTED_PREDICTIONS(source, destination_root)


_IMPL._selected_prediction_copy_target = _selected_prediction_copy_target
_IMPL._copy_selected_predictions = _copy_selected_predictions

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_selected_prediction_copy_target"] = _selected_prediction_copy_target
globals()["_copy_selected_predictions"] = _copy_selected_predictions

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
