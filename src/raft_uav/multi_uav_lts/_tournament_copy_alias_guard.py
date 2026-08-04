"""Prevent guarded-tournament materialization from deleting its source."""

from __future__ import annotations

from functools import wraps
from pathlib import Path

from . import tournament as _IMPL

_INSTALLED_ATTR = "_raft_uav_copy_alias_guard_installed"


def _copy_target(path: Path, output_dir: Path) -> Path:
    if path.is_dir():
        return output_dir / "selected_predictions"
    suffix = "".join(path.suffixes) or ".bin"
    return output_dir / f"selected_predictions{suffix}"


def install() -> None:
    """Install the selected-candidate copy guard once."""

    if getattr(_IMPL, _INSTALLED_ATTR, False):
        return
    original = _IMPL._copy_selected_predictions

    @wraps(original)
    def _copy_selected_predictions(path: Path, output_dir: Path) -> None:
        source = Path(path)
        destination = _copy_target(source, Path(output_dir))
        source_resolved = source.resolve()
        destination_resolved = destination.resolve()
        if source_resolved == destination_resolved or source_resolved.is_relative_to(
            destination_resolved
        ):
            raise ValueError(
                "selected candidate must not be the selected_predictions copy target "
                "or live inside it"
            )
        original(source, Path(output_dir))

    _IMPL._copy_selected_predictions = _copy_selected_predictions
    setattr(_IMPL, _INSTALLED_ATTR, True)


install()
