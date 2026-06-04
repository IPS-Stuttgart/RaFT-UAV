"""RaFT-UAV adapters for PyRecEst innovation/NIS diagnostics.

RaFT-UAV keeps source-specific RF/radar policy and CSV naming here, while
PyRecEst owns the generic NIS, gate, residual, and summary diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
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
except (ImportError, ModuleNotFoundError):

    @dataclass(frozen=True)
    class InnovationDiagnostic:
        nis: float | None
        residual_norm: float | None
        measurement_dim: int
        accepted: bool
        gate_threshold: float | None = None
        source: str | None = None
        action: str | None = None
        time: float | None = None
        metadata: Mapping[str, Any] | None = None

    @dataclass(frozen=True)
    class InnovationSummary:
        group: str
        count: int
        accepted_count: int
        rejected_count: int
        nis_mean: float | None

    def normalized_innovation_squared(residual: np.ndarray, covariance: np.ndarray) -> float:
        residual = np.asarray(residual, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)
        return float(residual @ np.linalg.pinv(covariance) @ residual)

    def chi_square_gate_threshold(measurement_dim: int, gate_probability: float = 0.95) -> float:
        try:
            from scipy.stats import chi2

            return float(chi2.ppf(float(gate_probability), int(measurement_dim)))
        except Exception:
            return float("inf")

    def innovation_diagnostic(
        *,
        residual: np.ndarray,
        covariance: np.ndarray,
        gate_threshold: float | None = None,
        gate_probability: float | None = None,
        source: str | None = None,
        action: str | None = None,
        time: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> InnovationDiagnostic:
        residual = np.asarray(residual, dtype=float).reshape(-1)
        threshold = gate_threshold
        if threshold is None and gate_probability is not None:
            threshold = chi_square_gate_threshold(residual.size, gate_probability)
        nis = normalized_innovation_squared(residual, covariance)
        return InnovationDiagnostic(
            nis=nis,
            residual_norm=float(np.linalg.norm(residual)),
            measurement_dim=int(residual.size),
            accepted=bool(threshold is None or nis <= threshold),
            gate_threshold=threshold,
            source=source,
            action=action,
            time=time,
            metadata=metadata,
        )

    def linear_innovation_diagnostic(
        *,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
        gate_threshold: float | None = None,
        gate_probability: float | None = None,
        source: str | None = None,
        action: str | None = None,
        time: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> InnovationDiagnostic:
        residual = np.asarray(measurement, dtype=float).reshape(-1) - np.asarray(
            measurement_matrix, dtype=float
        ) @ np.asarray(mean, dtype=float).reshape(-1)
        innovation_covariance = (
            np.asarray(measurement_matrix, dtype=float)
            @ np.asarray(covariance, dtype=float)
            @ np.asarray(measurement_matrix, dtype=float).T
            + np.asarray(measurement_covariance, dtype=float)
        )
        return innovation_diagnostic(
            residual=residual,
            covariance=innovation_covariance,
            gate_threshold=gate_threshold,
            gate_probability=gate_probability,
            source=source,
            action=action,
            time=time,
            metadata=metadata,
        )

    def diagnostic_from_record(record: Mapping[str, Any]) -> InnovationDiagnostic:
        return InnovationDiagnostic(
            nis=None if record.get("nis") is None else float(record["nis"]),
            residual_norm=None
            if record.get("residual_norm_m") is None
            else float(record["residual_norm_m"]),
            measurement_dim=int(record.get("measurement_dim", 0)),
            accepted=bool(record.get("accepted", True)),
            gate_threshold=None
            if record.get("gate_threshold") is None
            else float(record["gate_threshold"]),
            source=None if record.get("source") is None else str(record["source"]),
            action=None if record.get("update_action") is None else str(record["update_action"]),
            time=None if record.get("time_s") is None else float(record["time_s"]),
            metadata=None,
        )

    def diagnostics_from_records(records: Iterable[Mapping[str, Any]]) -> list[InnovationDiagnostic]:
        return [diagnostic_from_record(record) for record in records]

    def diagnostics_to_dicts(diagnostics: Iterable[InnovationDiagnostic]) -> list[dict[str, object]]:
        return [
            {
                "source": item.source,
                "accepted": item.accepted,
                "measurement_dim": item.measurement_dim,
                "nis": item.nis,
                "residual_norm_m": item.residual_norm,
            }
            for item in diagnostics
        ]

    def summarize_innovation_diagnostics(
        diagnostics: Iterable[InnovationDiagnostic],
    ) -> list[InnovationSummary]:
        grouped: dict[str, list[InnovationDiagnostic]] = {}
        for item in diagnostics:
            grouped.setdefault(item.source or "all", []).append(item)
        summaries = []
        for group, items in grouped.items():
            nis_values = [item.nis for item in items if item.nis is not None]
            summaries.append(
                InnovationSummary(
                    group=group,
                    count=len(items),
                    accepted_count=sum(1 for item in items if item.accepted),
                    rejected_count=sum(1 for item in items if not item.accepted),
                    nis_mean=float(np.mean(nis_values)) if nis_values else None,
                )
            )
        return summaries

    def summaries_to_dicts(summaries: Iterable[InnovationSummary]) -> list[dict[str, object]]:
        return [summary.__dict__.copy() for summary in summaries]

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
