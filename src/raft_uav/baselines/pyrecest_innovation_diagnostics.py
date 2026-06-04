"""RaFT-UAV adapters for PyRecEst innovation/NIS diagnostics.

RaFT-UAV keeps source-specific RF/radar policy and CSV naming here, while
PyRecEst owns the generic NIS, gate, residual, and summary diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from pyrecest.tracking import (
    InnovationDiagnostic,
    InnovationSummary,
    chi_square_gate_threshold,
    diagnostic_from_record,
    diagnostics_from_records,
    diagnostics_to_dicts,
    innovation_diagnostic,
    linear_innovation_diagnostic,
    normalized_innovation_squared,
    summaries_to_dicts,
    summarize_innovation_diagnostics,
)

__all__ = [
    "InnovationDiagnostic",
    "InnovationSummary",
    "chi_square_gate_threshold",
    "diagnostic_from_record",
    "diagnostics_from_records",
    "diagnostics_to_dicts",
    "innovation_diagnostic",
    "linear_innovation_diagnostic",
    "normalized_innovation_squared",
    "raft_linear_innovation_diagnostic",
    "raft_innovation_diagnostic_record",
    "summaries_to_dicts",
    "summarize_innovation_diagnostics",
    "summarize_raft_innovation_records",
]


def raft_linear_innovation_diagnostic(
    *,
    mean: np.ndarray,
    covariance_matrix: np.ndarray,
    measurement_vector: np.ndarray,
    observation_matrix: np.ndarray,
    measurement_covariance: np.ndarray,
    gate_threshold: float | None = None,
    gate_probability: float | None = None,
    source: str | None = None,
    action: str | None = None,
    time_s: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InnovationDiagnostic:
    """Return PyRecEst linear innovation diagnostics with RaFT metadata."""

    return linear_innovation_diagnostic(
        mean=mean,
        covariance=covariance_matrix,
        measurement=measurement_vector,
        measurement_matrix=observation_matrix,
        measurement_covariance=measurement_covariance,
        gate_threshold=gate_threshold,
        gate_probability=gate_probability,
        source=source,
        action=action,
        time=time_s,
        metadata=metadata,
    )


def raft_innovation_diagnostic_record(diagnostic: InnovationDiagnostic) -> dict[str, object]:
    """Return a compact RaFT-style diagnostic row for CSV/JSON outputs."""

    return {
        "time_s": diagnostic.time,
        "source": diagnostic.source,
        "measurement_dim": int(diagnostic.measurement_dim),
        "accepted": diagnostic.accepted,
        "update_action": diagnostic.action,
        "nis": diagnostic.nis,
        "gate_threshold": diagnostic.gate_threshold,
        "residual_norm_m": diagnostic.residual_norm,
        "mahalanobis_distance": None
        if diagnostic.nis is None
        else float(np.sqrt(max(0.0, diagnostic.nis))),
    }


def summarize_raft_innovation_records(
    records: Iterable[Mapping[str, Any]],
    *,
    source: str | None = None,
) -> list[InnovationSummary]:
    """Summarize RaFT tracking records through PyRecEst diagnostics."""

    diagnostics = diagnostics_from_records(records)
    if source is not None:
        diagnostics = [item for item in diagnostics if item.source == source]
    return summarize_innovation_diagnostics(diagnostics)
