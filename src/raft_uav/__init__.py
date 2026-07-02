"""Radar-RF Fusion Tracking for UAVs."""

import os
import sys
from collections.abc import Callable
from typing import Any

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


def _track5_trajectory_smooth_guard_install() -> Callable[[], None]:
    from raft_uav.mmuad import track5_trajectory_smooth

    original_main = track5_trajectory_smooth.main

    def install() -> None:
        def main(argv: list[str] | None = None) -> int:
            args = list(sys.argv[1:] if argv is None else argv)
            has_readiness_check = "--require-leaderboard-ready" in args
            has_template = "--template" in args or any(
                token.startswith("--template=") for token in args
            )
            if has_readiness_check and not has_template:
                raise SystemExit("--require-leaderboard-ready requires --template for validation")
            return original_main(argv)

        track5_trajectory_smooth.main = main

    return install


if os.environ.get("RAFT_UAV_SKIP_RUNTIME_HOOKS") != "1":
    _optional_runtime_hook(_radar_covariance_install)
    _optional_runtime_hook(_tracklet_viterbi_install)
    _optional_runtime_hook(_runtime_cli_patch_install)
    _optional_runtime_hook(_track5_trajectory_smooth_guard_install)
