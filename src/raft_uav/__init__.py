"""Radar-RF Fusion Tracking for UAVs."""

import os
from collections.abc import Callable

os.environ.setdefault("MPLBACKEND", "Agg")

__all__ = ["__version__"]

__version__ = "0.1.0"


def _optional_runtime_hook(import_install: Callable[[], Callable[[], None]]) -> None:
    """Install startup hooks when their optional runtime dependencies exist."""

    try:
        install = import_install()
        install()
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name.startswith("pyrecest"):
            return
        raise


def _radar_covariance_install() -> Callable[[], None]:
    from raft_uav.baselines.radar_covariance_runtime import install

    return install


def _tracklet_viterbi_install() -> Callable[[], None]:
    from raft_uav.baselines.tracklet_viterbi_runtime import install

    return install


def _runtime_cli_patch_install() -> Callable[[], None]:
    from raft_uav.runtime_cli_patch import install

    return install


_optional_runtime_hook(_radar_covariance_install)
_optional_runtime_hook(_tracklet_viterbi_install)
_optional_runtime_hook(_runtime_cli_patch_install)
