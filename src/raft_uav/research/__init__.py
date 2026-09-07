"""Research utilities for higher-ceiling RaFT-UAV experiments.

The modules in this package are intentionally lightweight and dataset-agnostic.
They provide reusable building blocks for diagnostics, calibration, smoothing,
association repair, and reproducibility workflows without changing the default
tracking pipeline.
"""

from raft_uav.research import diagnostics as _diagnostics
from raft_uav.research import factor_graph as _factor_graph
from raft_uav.research._diagnostics_distance_stability_patch import (
    apply_diagnostics_distance_stability_patch,
)
from raft_uav.research._diagnostics_domain_shift_stability_patch import (
    apply_diagnostics_domain_shift_stability_patch,
)
from raft_uav.research._diagnostics_latency_stability_patch import (
    apply_diagnostics_latency_stability_patch,
)
from raft_uav.research._diagnostics_numeric_frame_patch import (
    apply_diagnostics_numeric_frame_patch,
)
from raft_uav.research._diagnostics_numeric_time_patch import (
    apply_diagnostics_numeric_time_patch,
)
from raft_uav.research._factor_graph_frame_group_patch import (
    apply_factor_graph_frame_group_patch,
)
from raft_uav.research._factor_graph_sequence_guard_patch import (
    apply_factor_graph_sequence_guard_patch,
)
from raft_uav.research.uncertainty import ConformalRadius, fit_conformal_radius

apply_diagnostics_numeric_time_patch(_diagnostics)
apply_diagnostics_numeric_frame_patch(_diagnostics)
apply_diagnostics_domain_shift_stability_patch(_diagnostics)
apply_diagnostics_distance_stability_patch(_diagnostics)
apply_diagnostics_latency_stability_patch(_diagnostics)
apply_factor_graph_frame_group_patch(_factor_graph)
apply_factor_graph_sequence_guard_patch(_factor_graph)

association_regret = _diagnostics.association_regret
association_regret_summary = _diagnostics.association_regret_summary
candidate_set_recall = _diagnostics.candidate_set_recall
domain_shift_summary = _diagnostics.domain_shift_summary
latency_curve = _diagnostics.latency_curve
leakage_sentinel = _diagnostics.leakage_sentinel
track_switch_metrics = _diagnostics.track_switch_metrics

FactorGraphSmoothingResult = _factor_graph.FactorGraphSmoothingResult
LeastSquaresSmoothingConfig = _factor_graph.LeastSquaresSmoothingConfig
coordinate_descent_association_and_smoothing = (
    _factor_graph.coordinate_descent_association_and_smoothing
)
smooth_position_trajectory = _factor_graph.smooth_position_trajectory

__all__ = [
    "ConformalRadius",
    "FactorGraphSmoothingResult",
    "LeastSquaresSmoothingConfig",
    "association_regret",
    "association_regret_summary",
    "candidate_set_recall",
    "coordinate_descent_association_and_smoothing",
    "domain_shift_summary",
    "fit_conformal_radius",
    "latency_curve",
    "leakage_sentinel",
    "smooth_position_trajectory",
    "track_switch_metrics",
]
