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
        if missing_name == "pyrecest" or missing_name.startswith("pyrecest."):
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


def _kalman_timestamp_validation_install() -> Callable[[], None]:
    from raft_uav.baselines._kalman_timestamp_validation_patch import (
        apply_kalman_timestamp_validation_patch,
    )

    return apply_kalman_timestamp_validation_patch


def _rf_measurement_fallback_install() -> Callable[[], None]:
    from raft_uav.io._rf_measurement_fallback_patch import (
        apply_rf_measurement_fallback_patch,
    )

    return apply_rf_measurement_fallback_patch


def _radar_measurement_validation_install() -> Callable[[], None]:
    from raft_uav.io._radar_measurement_validation_patch import (
        apply_radar_measurement_validation_patch,
    )

    return apply_radar_measurement_validation_patch


def _catprob_sequence_install() -> Callable[[], None]:
    from raft_uav.io._catprob_sequence_patch import install

    return install


def _uncertainty_payload_validation_install() -> Callable[[], None]:
    from raft_uav._uncertainty_payload_validation_patch import install

    return install


def _candidate_reservoir_config_validation_install() -> Callable[[], None]:
    from raft_uav.mmuad._candidate_reservoir_config_validation_patch import install

    return install


def _acceleration_limit_displacement_install() -> Callable[[], None]:
    from raft_uav.mmuad._acceleration_limit_displacement_patch import install

    return install


def _repair_complex_row_validation_install() -> Callable[[], None]:
    from raft_uav.mmuad._repair_complex_row_validation_patch import install

    return install


if os.environ.get("RAFT_UAV_SKIP_RUNTIME_HOOKS") != "1":
    _optional_runtime_hook(_radar_covariance_install)
    _optional_runtime_hook(_tracklet_viterbi_install)
    _optional_runtime_hook(_runtime_cli_patch_install)

# Scalar validation and normalized measurement fallbacks are core input-safety
# boundaries, not optional integrations. Keep them active when runtime hooks are skipped.
_optional_runtime_hook(_kalman_timestamp_validation_install)
_optional_runtime_hook(_rf_measurement_fallback_install)
_optional_runtime_hook(_radar_measurement_validation_install)
_optional_runtime_hook(_catprob_sequence_install)
_optional_runtime_hook(_uncertainty_payload_validation_install)
_optional_runtime_hook(_candidate_reservoir_config_validation_install)
_optional_runtime_hook(_acceleration_limit_displacement_install)
_optional_runtime_hook(_repair_complex_row_validation_install)
