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


def _imm_bootstrap_timestamp_install() -> Callable[[], None]:
    from raft_uav.baselines._imm_bootstrap_timestamp_patch import install

    return install


def _imm_cli_numeric_validation_install() -> Callable[[], None]:
    from raft_uav._imm_cli_numeric_validation_patch import install

    return install


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


def _radar_velocity_strict_mode_install() -> Callable[[], None]:
    from raft_uav.io._radar_velocity_strict_mode_patch import install

    return install


def _geodetic_input_validation_install() -> Callable[[], None]:
    from raft_uav.io._geodetic_input_validation_patch import install

    return install


def _catprob_sequence_install() -> Callable[[], None]:
    from raft_uav.io._catprob_sequence_patch import install

    return install


def _uncertainty_payload_validation_install() -> Callable[[], None]:
    from raft_uav._uncertainty_payload_validation_patch import install

    return install


def _uncertainty_apply_validation_install() -> Callable[[], None]:
    from raft_uav._uncertainty_apply_validation_patch import install

    return install


def _uncertainty_duplicate_truth_install() -> Callable[[], None]:
    from raft_uav._uncertainty_duplicate_truth_patch import install

    return install


def _radar_text_id_csv_install() -> Callable[[], None]:
    from raft_uav.mmuad._radar_text_id_csv_patch import install

    return install


def _point_cloud_source_name_install() -> Callable[[], None]:
    from raft_uav.mmuad._point_cloud_source_name_patch import install

    return install


def _binary_ply_native_endian_install() -> Callable[[], None]:
    from raft_uav.mmuad._binary_ply_native_endian_patch import install

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


def _jerk_window_support_install() -> Callable[[], None]:
    from raft_uav.mmuad._jerk_window_support_patch import install

    return install


def _acceleration_sequence_id_install() -> Callable[[], None]:
    from raft_uav.mmuad._acceleration_sequence_id_patch import install

    return install


def _mot_match_distance_complex_install() -> Callable[[], None]:
    from raft_uav.mmuad._mot_match_distance_complex_patch import install

    return install


def _mot_config_validation_install() -> Callable[[], None]:
    from raft_uav.mmuad._mot_config_validation_patch import install

    return install


def _mot_missing_id_index_install() -> Callable[[], None]:
    from raft_uav.mmuad._mot_missing_id_index_patch import install

    return install


def _candidate_pull_index_install() -> Callable[[], None]:
    from raft_uav.mmuad._candidate_pull_index_patch import install

    return install


def _candidate_pull_time_tolerance_install() -> Callable[[], None]:
    from raft_uav.mmuad._candidate_pull_time_tolerance_patch import install

    return install


def _radar_frame_grouping_install() -> Callable[[], None]:
    from raft_uav.baselines._radar_frame_grouping_patch import install

    return install


def _paper_selection_numeric_time_install() -> Callable[[], None]:
    from raft_uav._paper_selection_numeric_time_patch import install

    return install


def _oracle_coverage_sequence_scope_install() -> Callable[[], None]:
    from raft_uav.evaluation._oracle_coverage_sequence_scope_patch import install

    return install


def _estimate_sequence_apply_coverage_install() -> Callable[[], None]:
    from raft_uav.mmuad._estimate_sequence_apply_coverage_patch import install

    return install


def _track5_rts_template_validation_install() -> Callable[[], None]:
    from raft_uav.mmuad._track5_rts_template_validation_patch import install

    return install


def _geomedian_template_time_match_install() -> Callable[[], None]:
    from raft_uav.mmuad._geomedian_template_time_match_patch import install

    return install


def _candidate_score_calibration_order_install() -> Callable[[], None]:
    from raft_uav.mmuad._candidate_score_calibration_order_patch import install

    return install


def _track5_submission_schema_guard_install() -> Callable[[], None]:
    from raft_uav.mmuad._track5_submission_schema_guard_patch import install

    return install


def _track5_estimate_calibration_duplicate_truth_install() -> Callable[[], None]:
    from raft_uav.mmuad._track5_estimate_calibration_duplicate_truth_patch import install

    return install


def _track5_template_resample_template_validation_install() -> Callable[[], None]:
    from raft_uav.mmuad._track5_template_resample_template_validation_patch import install

    return install


def _evaluate_fde_truth_identity_install() -> Callable[[], None]:
    from raft_uav.mmuad._evaluate_fde_truth_identity_patch import install

    return install


def _evaluate_flight_scope_install() -> Callable[[], None]:
    from raft_uav.mmuad._evaluate_flight_scope_patch import install

    return install


def _trajectory_completion_truth_scope_install() -> Callable[[], None]:
    from raft_uav.mmuad._trajectory_completion_truth_scope_patch import install

    return install


if os.environ.get("RAFT_UAV_SKIP_RUNTIME_HOOKS") != "1":
    _optional_runtime_hook(_radar_covariance_install)
    _optional_runtime_hook(_tracklet_viterbi_install)
    _optional_runtime_hook(_runtime_cli_patch_install)

# Scalar validation and normalized measurement fallbacks are core input-safety
# boundaries, not optional integrations. Keep them active when runtime hooks are skipped.
_optional_runtime_hook(_kalman_timestamp_validation_install)
_optional_runtime_hook(_imm_bootstrap_timestamp_install)
_optional_runtime_hook(_imm_cli_numeric_validation_install)
_optional_runtime_hook(_rf_measurement_fallback_install)
_optional_runtime_hook(_radar_measurement_validation_install)
_optional_runtime_hook(_radar_velocity_strict_mode_install)
_optional_runtime_hook(_geodetic_input_validation_install)
_optional_runtime_hook(_catprob_sequence_install)
_optional_runtime_hook(_uncertainty_payload_validation_install)
_optional_runtime_hook(_uncertainty_apply_validation_install)
_optional_runtime_hook(_uncertainty_duplicate_truth_install)
_optional_runtime_hook(_radar_text_id_csv_install)
_optional_runtime_hook(_point_cloud_source_name_install)
_optional_runtime_hook(_binary_ply_native_endian_install)
_optional_runtime_hook(_candidate_reservoir_config_validation_install)
_optional_runtime_hook(_acceleration_limit_displacement_install)
_optional_runtime_hook(_repair_complex_row_validation_install)
_optional_runtime_hook(_jerk_window_support_install)
_optional_runtime_hook(_acceleration_sequence_id_install)
_optional_runtime_hook(_mot_match_distance_complex_install)
_optional_runtime_hook(_mot_config_validation_install)
_optional_runtime_hook(_mot_missing_id_index_install)
_optional_runtime_hook(_candidate_pull_index_install)
_optional_runtime_hook(_candidate_pull_time_tolerance_install)
_optional_runtime_hook(_radar_frame_grouping_install)
_optional_runtime_hook(_paper_selection_numeric_time_install)
_optional_runtime_hook(_oracle_coverage_sequence_scope_install)
_optional_runtime_hook(_estimate_sequence_apply_coverage_install)
_optional_runtime_hook(_track5_rts_template_validation_install)
_optional_runtime_hook(_geomedian_template_time_match_install)
_optional_runtime_hook(_candidate_score_calibration_order_install)
_optional_runtime_hook(_track5_submission_schema_guard_install)
_optional_runtime_hook(_track5_estimate_calibration_duplicate_truth_install)
_optional_runtime_hook(_track5_template_resample_template_validation_install)
_optional_runtime_hook(_evaluate_fde_truth_identity_install)
_optional_runtime_hook(_evaluate_flight_scope_install)
_optional_runtime_hook(_trajectory_completion_truth_scope_install)
