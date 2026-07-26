"""Baseline trackers."""

from raft_uav.baselines import imm as _imm
from raft_uav.baselines import pyrecest_innovation_diagnostics as _pyrecest_innovation_diagnostics
from raft_uav.baselines import radar_association as _radar_association
from raft_uav.baselines import smoothing as _smoothing
from raft_uav.baselines._imm_transition_validation_patch import (
    apply_imm_transition_validation_patch,
)
from raft_uav.baselines._innovation_diagnostic_record_patch import (
    apply_innovation_diagnostic_record_patch,
)
from raft_uav.baselines._radar_association_interpolation_patch import (
    apply_radar_association_interpolation_patch,
)
from raft_uav.baselines._robust_map_accepted_matching_patch import (
    apply_robust_map_accepted_matching_patch,
)
from raft_uav.baselines._robust_map_lag_validation_patch import (
    apply_robust_map_lag_validation_patch,
)

apply_imm_transition_validation_patch(_imm)
apply_innovation_diagnostic_record_patch(_pyrecest_innovation_diagnostics)

AsyncInteractingMultipleModelTracker = _imm.AsyncInteractingMultipleModelTracker
IMMMode = _imm.IMMMode
default_imm_modes = _imm.default_imm_modes
fixed_turn_rate_matrix = _imm.fixed_turn_rate_matrix
run_async_imm_baseline = _imm.run_async_imm_baseline
uniform_ctmc_transition_matrix = _imm.uniform_ctmc_transition_matrix

apply_radar_association_interpolation_patch()
apply_robust_map_accepted_matching_patch()
apply_robust_map_lag_validation_patch()

RADAR_ASSOCIATION_MODES = _radar_association.RADAR_ASSOCIATION_MODES
run_async_cv_baseline_with_radar_association = (
    _radar_association.run_async_cv_baseline_with_radar_association
)
SMOOTHER_MODES = _smoothing.SMOOTHER_MODES
smooth_tracking_records = _smoothing.smooth_tracking_records

__all__ = [
    "AsyncInteractingMultipleModelTracker",
    "IMMMode",
    "RADAR_ASSOCIATION_MODES",
    "SMOOTHER_MODES",
    "default_imm_modes",
    "fixed_turn_rate_matrix",
    "run_async_cv_baseline_with_radar_association",
    "run_async_imm_baseline",
    "smooth_tracking_records",
    "uniform_ctmc_transition_matrix",
]
