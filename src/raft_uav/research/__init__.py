"""Research utilities for higher-ceiling RaFT-UAV experiments.

The modules in this package are intentionally lightweight and dataset-agnostic.
They provide reusable building blocks for diagnostics, calibration, smoothing,
association repair, and reproducibility workflows without changing the default
tracking pipeline.
"""

from raft_uav.research.diagnostics import (
    association_regret,
    association_regret_summary,
    candidate_set_recall,
    domain_shift_summary,
    latency_curve,
    leakage_sentinel,
    track_switch_metrics,
)
from raft_uav.research.factor_graph import (
    FactorGraphSmoothingResult,
    LeastSquaresSmoothingConfig,
    coordinate_descent_association_and_smoothing,
    smooth_position_trajectory,
)
from raft_uav.research.uncertainty import ConformalRadius, fit_conformal_radius

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
