"""Tie-invariant score and posterior ranks for adaptive pair priors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_scores_with_average_ties(values: np.ndarray, mode: str) -> np.ndarray:
    """Normalize finite scores without using row order to break rank ties."""

    score = np.asarray(values, dtype=float)
    finite = np.isfinite(score)
    if not finite.any():
        return np.zeros_like(score, dtype=float)
    floor = float(np.min(score[finite]))
    score = np.where(finite, score, floor)
    if mode == "none":
        return score
    if mode == "rank":
        if len(score) <= 1:
            return np.full(len(score), 0.5, dtype=float)
        ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
        return (ranks - 1.0) / float(len(score) - 1)
    minimum = float(np.min(score))
    maximum = float(np.max(score))
    if maximum <= minimum:
        return np.full(len(score), 0.5, dtype=float)
    return (score - minimum) / (maximum - minimum)


def descending_average_ranks(values: np.ndarray) -> np.ndarray:
    """Return descending ranks that assign equal average ranks to ties."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    finite = np.isfinite(array)
    ranks = np.empty(array.shape, dtype=float)
    if not finite.any():
        ranks.fill(1.0)
        return ranks
    ranks[finite] = (
        pd.Series(array[finite])
        .rank(method="average", ascending=False)
        .to_numpy(dtype=float)
    )
    ranks[~finite] = float(np.sum(finite) + 1)
    return ranks
