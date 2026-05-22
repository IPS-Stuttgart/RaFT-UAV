"""Soft output fusion for weighted tracking hypotheses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyrecest.filters.gaussian_hypothesis_mixture import (
    WeightedGaussianHypothesis as GaussianHypothesis,
    moment_match_gaussian_hypotheses as moment_match_hypotheses,
    normalize_log_weights,
)


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
    scores = pd.to_numeric(rows.get(score_column, pd.Series(0.0, index=rows.index)), errors="coerce")
    log_weights = -scores.fillna(scores.max() if scores.notna().any() else 0.0).to_numpy(dtype=float)
    has_covariance = all(column in rows.columns for column in covariance_columns)
    hypotheses = []
    for index, (_, row) in enumerate(rows.iterrows()):
        row_covariance = (
            _covariance_from_row(row, covariance_columns)
            if has_covariance
            else np.diag([25.0**2, 25.0**2, 35.0**2])
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


def _covariance_from_row(row: pd.Series, columns: tuple[str, str, str, str, str, str]) -> np.ndarray:
    ee, nn, uu, en, eu, nu = [float(row[column]) for column in columns]
    covariance = np.array([[ee, en, eu], [en, nn, nu], [eu, nu, uu]], dtype=float)
    if not np.isfinite(covariance).all():
        return np.diag([25.0**2, 25.0**2, 35.0**2])
    return _symmetrized(covariance)


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
