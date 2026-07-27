"""Reject malformed explicit multi-object tracker configurations."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

_PATCH_MARKER = "_raft_uav_validates_explicit_mot_config"


def install() -> None:
    """Install strict configuration validation at both public MOT entry points."""

    import raft_uav.mmuad as mmuad
    from raft_uav.mmuad import mot

    original: Callable[..., Any] = mot.run_mmuad_multi_object_tracker
    if getattr(original, _PATCH_MARKER, False):
        setattr(mmuad, "run_mmuad_multi_object_tracker", original)
        return

    @wraps(original)
    def validated(
        candidates: Any,
        truth: Any = None,
        *,
        config: Any = None,
    ) -> Any:
        if config is not None and not isinstance(config, mot.MultiObjectTrackerConfig):
            raise TypeError("config must be a MultiObjectTrackerConfig or None")
        return original(candidates, truth, config=config)

    setattr(validated, _PATCH_MARKER, True)
    setattr(mot, "run_mmuad_multi_object_tracker", validated)
    setattr(mmuad, "run_mmuad_multi_object_tracker", validated)
