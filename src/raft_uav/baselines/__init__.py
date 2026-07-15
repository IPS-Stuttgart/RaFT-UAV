"""Baseline trackers."""

from raft_uav.baselines import imm as _imm
from raft_uav.baselines import radar_association as _radar_association
from raft_uav.baselines._imm_transition_validation_patch import (
    apply_imm_transition_validation_patch,
)
from raft_uav.baselines._radar_association_interpolation_patch import (
    apply_radar_association_interpolation_patch,
)
from raft_uav.baselines._robust_map_accepted_matching_patch import (
    apply_robust_map_accepted_matching_patch,
)

apply_imm_transition_validation_patch(_imm)

from raft_uav.baselines.imm import (  # noqa: E402
    AsyncInteractingMultipleModelTracker,
    IMMMode,
    default_imm_modes,
    fixed_turn_rate_matrix,
    run_async_imm_baseline,
    uniform_ctmc_transition_matrix,
)
from raft_uav.baselines.smoothing import SMOOTHER_MODES, smooth_tracking_records

apply_radar_association_interpolation_patch()
apply_robust_map_accepted_matching_patch()

RADAR_ASSOCIATION_MODES = _radar_association.RADAR_ASSOCIATION_MODES
run_async_cv_baseline_with_radar_association = (
    _radar_association.run_async_cv_baseline_with_radar_association
)

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
