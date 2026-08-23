"""Delegate candidate-mixture factor evaluation to PyRecEst.

RaFT-UAV retains MMUAD candidate preparation, application-specific balancing,
and trajectory regularization. The reusable Gaussian-mixture likelihood,
responsibility, and moment-matching calculations live in PyRecEst.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class _ApplyLabelBalance(Protocol):
    def __call__(
        self,
        weights: np.ndarray,
        labels: np.ndarray,
        *,
        balance: float,
    ) -> np.ndarray: ...


class _NormalizeProbability(Protocol):
    def __call__(self, values: np.ndarray) -> np.ndarray: ...


def build_pyrecest_mixture_response(
    *,
    apply_label_balance: _ApplyLabelBalance,
    normalize_probability: _NormalizeProbability,
):
    """Build the legacy-compatible response function backed by PyRecEst."""

    def _mixture_response(
        frames: Sequence[dict[str, Any]],
        state: np.ndarray,
        *,
        config: Any,
    ) -> list[dict[str, Any]]:
        factor_type = _load_factor_type()
        response: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            positions = np.asarray(frame["positions"], dtype=float)
            sigmas = np.asarray(frame["sigmas"], dtype=float)
            scores = np.asarray(frame["normalized_scores"], dtype=float)
            measurement_dim = int(positions.shape[1])
            sigma_log_weight = float(config.sigma_log_weight)

            if sigma_log_weight == 0.0:
                covariances = np.eye(measurement_dim, dtype=float)
                log_determinant_weight = 0.0
            else:
                identity = np.eye(measurement_dim, dtype=float)
                covariances = sigmas[:, None, None] ** 2 * identity[None, :, :]
                log_determinant_weight = sigma_log_weight / float(measurement_dim)

            factor = factor_type(
                means=positions,
                covariances=covariances,
                log_weights=(
                    float(config.score_weight) * scores / float(config.temperature)
                ),
                loss=config.loss,
                huber_delta=float(config.huber_delta),
                log_determinant_weight=log_determinant_weight,
            )
            evaluation = factor.evaluate(np.asarray(state[frame_index], dtype=float))
            weights = np.asarray(evaluation.responsibilities, dtype=float).copy()

            floor = float(config.uniform_weight_floor)
            if floor > 0.0:
                weights = (1.0 - floor) * weights + floor / len(weights)

            rows = frame["rows"]
            weights = apply_label_balance(
                weights,
                rows["candidate_branch"].astype(str).to_numpy()
                if "candidate_branch" in rows
                else np.full(len(weights), "unknown", dtype=object),
                balance=float(config.branch_balance),
            )
            weights = apply_label_balance(
                weights,
                rows["source"].astype(str).to_numpy()
                if "source" in rows
                else np.full(len(weights), "unknown", dtype=object),
                balance=float(config.source_balance),
            )

            floor = float(config.responsibility_floor)
            if floor > 0.0:
                weights = (1.0 - floor) * weights + floor / len(weights)
            weights = normalize_probability(weights)

            pseudo_position, effective_covariance = factor.moment_match(weights)
            effective_variance = max(
                float(np.trace(effective_covariance)) / float(measurement_dim),
                1.0e-12,
            )
            precision = float(
                np.clip(
                    1.0 / effective_variance,
                    float(config.min_measurement_precision),
                    float(config.max_measurement_precision),
                )
            )
            entropy = float(
                -np.sum(weights * np.log(np.maximum(weights, 1.0e-300)))
            )
            response.append(
                {
                    "weights": weights,
                    "distances": np.linalg.norm(evaluation.residuals, axis=1),
                    "normalized_residual": evaluation.mahalanobis_distances,
                    "robust_cost": evaluation.robust_costs,
                    "log_weight": evaluation.component_log_weights,
                    "pseudo_position": pseudo_position,
                    "effective_sigma_m": float(np.sqrt(effective_variance)),
                    "measurement_precision": precision,
                    "entropy": entropy,
                    "effective_candidate_count": float(np.exp(entropy)),
                    "dominant_index": int(np.argmax(weights)),
                }
            )
        return response

    return _mixture_response


def _load_factor_type():
    try:
        from pyrecest.filters.candidate_mixture import (
            GaussianMixtureMeasurementFactor,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"pyrecest", "pyrecest.filters.candidate_mixture"}:
            raise
        raise ImportError(
            "candidate-mixture MAP requires a PyRecEst version providing "
            "pyrecest.filters.candidate_mixture"
        ) from exc
    return GaussianMixtureMeasurementFactor


__all__ = ["build_pyrecest_mixture_response"]
