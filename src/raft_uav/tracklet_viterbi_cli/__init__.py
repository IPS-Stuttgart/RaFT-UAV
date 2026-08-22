"""Compatibility fixes for canonical tracklet-Viterbi CLI dispatch.

The maintained implementation lives in the sibling ``tracklet_viterbi_cli.py``
module. This package preserves the public import and console-script surfaces
while keeping CLI-generated experiment overlays compatible with the hardened
base tracklet runner.
"""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_viterbi_cli.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav._tracklet_viterbi_cli_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load canonical tracklet CLI from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRACKLET_RUNNER_FROM_ENVIRONMENT = _IMPL._tracklet_runner_from_environment


def _base_config_from_overlay(config: Any) -> Any:
    """Unwrap the canonical CLI overlay only for the plain base runner.

    ``_TrackletConfigOverlay`` carries retention/range experiment-only fields,
    but its ``_base`` member is the exact ``TrackletViterbiAssociationConfig``
    expected by the hardened base runner. Non-overlay inputs are left untouched
    so the base runner keeps rejecting malformed explicit configurations.
    """

    if isinstance(config, _IMPL._TrackletConfigOverlay):
        return config._base
    return config


def _tracklet_runner_from_environment() -> Callable[..., Any]:
    """Return the selected runner with base-only config unwrapping."""

    variant = _IMPL._tracklet_variant_from_environment()
    runner = _ORIGINAL_TRACKLET_RUNNER_FROM_ENVIRONMENT()
    if variant != "base":
        return runner

    @wraps(runner)
    def run_base_with_cli_config(*args: Any, **kwargs: Any) -> Any:
        if "config" in kwargs:
            normalized_kwargs = dict(kwargs)
            normalized_kwargs["config"] = _base_config_from_overlay(kwargs["config"])
            kwargs = normalized_kwargs
        return runner(*args, **kwargs)

    return run_base_with_cli_config


_IMPL._tracklet_runner_from_environment = _tracklet_runner_from_environment

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_base_config_from_overlay"] = _base_config_from_overlay
globals()["_tracklet_runner_from_environment"] = _tracklet_runner_from_environment

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
