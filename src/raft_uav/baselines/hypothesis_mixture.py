"""Soft output fusion for weighted tracking hypotheses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyrecest.filters.gaussian_hypothesis_mixture import (
    WeightedGaussianHypothesis as GaussianHypothesis,
    moment_match_gaussian_hypotheses as moment_match_hypotheses,
)
from pyrecest.numerics import is_positive_semidefinite

from raft_uav.numeric import optional_float


_DEFAULT_POSITION_COVARIANCE = np.diag([25.0**2, 25.0**2, 35.0**2])


def position_mixture_from_association_rows(
    rows: pd.DataFrame,
    *,
    covariance_columns: tuple[str, str, str, str, str, str] = (
        "association_cov_ee",
        "association_cov_nn",
        "association_cov_uu",
        "association_cov_en",
        "association_cov_eu",
        "association_cov_nu",
    ),
    score_column: str = "association_score",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moment-match candidate rows into one soft position update.

    RaFT-UAV keeps the pandas/column conversion here and delegates Gaussian
    log-weight normalization and moment matching to PyRecEst.
    """

    if rows.empty:
        raise ValueError("rows must not be empty")
    positions = rows[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    log_weights = _association_log_weights(rows, score_column)
    has_covariance = all(column in rows.columns for column in covariance_columns)
    hypotheses = []
    for index, (_, row) in enumerate(rows.iterrows()):
        row_covariance = (
            _covariance_from_row(row, covariance_columns)
            if has_covariance
            else _DEFAULT_POSITION_COVARIANCE.copy()
        )
        hypotheses.append(
            GaussianHypothesis(
                mean=positions[index],
                covariance=row_covariance,
                log_weight=float(log_weights[index]),
                metadata={"row_index": int(index)},
            )
        )
    return moment_match_hypotheses(hypotheses)


def _association_log_weights(rows: pd.DataFrame, score_column: str) -> np.ndarray:
    """Return finite log weights, treating malformed scores as worst finite scores."""

    raw_scores = rows.get(score_column, pd.Series(0.0, index=rows.index))
    numeric_scores = pd.to_numeric(raw_scores, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric_scores)
    fallback_score = float(numeric_scores[finite].max()) if finite.any() else 0.0
    safe_scores = np.where(finite, numeric_scores, fallback_score)
    return -safe_scores


def _covariance_from_row(
    row: pd.Series,
    columns: tuple[str, str, str, str, str, str],
) -> np.ndarray:
    values: list[float] = []
    for column in columns:
        value = optional_float(row[column])
        if value is None:
            return _DEFAULT_POSITION_COVARIANCE.copy()
        values.append(value)
    ee, nn, uu, en, eu, nu = values
    covariance = _symmetrized(
        np.array([[ee, en, eu], [en, nn, nu], [eu, nu, uu]], dtype=float)
    )
    if not np.isfinite(covariance).all() or not is_positive_semidefinite(covariance):
        return _DEFAULT_POSITION_COVARIANCE.copy()
    return covariance


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
