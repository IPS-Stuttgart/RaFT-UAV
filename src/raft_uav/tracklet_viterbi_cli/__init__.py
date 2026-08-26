"""Compatibility fixes for canonical tracklet-Viterbi CLI dispatch.

The maintained implementation lives in the sibling ``tracklet_viterbi_cli.py``
module. This package preserves the public import and console-script surfaces
while keeping CLI-generated experiment overlays compatible with the hardened
base tracklet runner.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable

# Execute the maintained legacy module in *this* module namespace. Loading it
# into a second module and copying its attributes here looks equivalent, but is
# not: functions retain the globals dictionary of the module where they were
# defined. That split breaks callers and tests which monkeypatch the public
# ``raft_uav.tracklet_viterbi_cli`` module because the runner would continue to
# read stale globals from the hidden implementation module.
_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_viterbi_cli.py"
with _IMPL_PATH.open("rb") as _source:
    exec(compile(_source.read(), str(_IMPL_PATH), "exec"), globals(), globals())

_ORIGINAL_TRACKLET_RUNNER_FROM_ENVIRONMENT = globals()[
    "_tracklet_runner_from_environment"
]


def _base_config_from_overlay(config: Any) -> Any:
    """Unwrap the canonical CLI overlay only for the plain base runner.

    ``_TrackletConfigOverlay`` carries retention/range experiment-only fields,
    but its ``_base`` member is the exact ``TrackletViterbiAssociationConfig``
    expected by the hardened base runner. Non-overlay inputs are left untouched
    so the base runner keeps rejecting malformed explicit configurations.
    """

    if isinstance(config, globals()["_TrackletConfigOverlay"]):
        return config._base
    return config


def _tracklet_runner_from_environment() -> Callable[..., Any]:
    """Return the selected runner with base-only config unwrapping."""

    variant = globals()["_tracklet_variant_from_environment"]()
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
