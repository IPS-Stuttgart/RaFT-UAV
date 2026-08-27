"""Keep domain-shift statistics finite when finite inputs have extreme magnitude."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import Callable

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_domain_shift_stability_patch_applied"
_REPAIR_COLUMNS = (
    "train_mean",
    "heldout_mean",
    "mean_shift_z",
    "train_p50",
    "heldout_p50",
    "train_p90",
    "heldout_p90",
)


def _stable_distribution_statistics(values: np.ndarray) -> dict[str, float]:
    """Return mean, population standard deviation, and quantiles without overflow."""

    array = np.asarray(values, dtype=float).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
        }
    if all(np.isfinite(value) for value in direct.values()):
        return direct

    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return direct

    normalized = array / scale
    with np.errstate(over="ignore", invalid="ignore"):
        return {
            "mean": float(scale * np.mean(normalized)),
            "std": float(scale * np.std(normalized)),
            "p50": float(scale * np.percentile(normalized, 50)),
            "p90": float(scale * np.percentile(normalized, 90)),
        }


def _stable_mean_shift_z(
    train_values: np.ndarray,
    heldout_values: np.ndarray,
    *,
    train_statistics: dict[str, float],
    heldout_statistics: dict[str, float],
) -> float:
    """Return the standardized mean shift without overflowing the numerator."""

    train_std = train_statistics["std"]
    denominator = train_std if train_std != 0.0 else 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        direct = (
            heldout_statistics["mean"] - train_statistics["mean"]
        ) / denominator
    if np.isfinite(direct):
        return float(direct)

    scale = max(
        float(np.max(np.abs(train_values))),
        float(np.max(np.abs(heldout_values))),
    )
    if scale == 0.0:
        return 0.0

    normalized_train = train_values / scale
    normalized_heldout = heldout_values / scale
    normalized_shift = float(
        np.mean(normalized_heldout) - np.mean(normalized_train)
    )
    if train_std == 0.0:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(scale * normalized_shift)

    normalized_std = float(np.std(normalized_train))
    if normalized_std == 0.0:
        return float(direct)
    return float(normalized_shift / normalized_std)


def apply_diagnostics_domain_shift_stability_patch(module: ModuleType) -> None:
    """Patch ``domain_shift_summary`` while preserving ordinary outputs."""

    original: Callable[..., pd.DataFrame] = module.domain_shift_summary
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def domain_shift_summary(
        training: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame] | pd.DataFrame,
        heldout: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = original(training, heldout, columns=columns)
        if result.empty or np.isfinite(
            result.loc[:, list(_REPAIR_COLUMNS)].to_numpy(dtype=float)
        ).all():
            return result

        train = module._concat_training_frames(training)
        repaired = result.copy()
        for index, feature in repaired["feature"].items():
            values = repaired.loc[index, list(_REPAIR_COLUMNS)].to_numpy(dtype=float)
            if np.isfinite(values).all():
                continue

            train_values = pd.to_numeric(
                train[feature], errors="coerce"
            ).to_numpy(dtype=float)
            heldout_values = pd.to_numeric(
                heldout[feature], errors="coerce"
            ).to_numpy(dtype=float)
            train_values = train_values[np.isfinite(train_values)]
            heldout_values = heldout_values[np.isfinite(heldout_values)]

            train_statistics = _stable_distribution_statistics(train_values)
            heldout_statistics = _stable_distribution_statistics(heldout_values)
            repaired.at[index, "train_mean"] = train_statistics["mean"]
            repaired.at[index, "heldout_mean"] = heldout_statistics["mean"]
            repaired.at[index, "mean_shift_z"] = _stable_mean_shift_z(
                train_values,
                heldout_values,
                train_statistics=train_statistics,
                heldout_statistics=heldout_statistics,
            )
            repaired.at[index, "train_p50"] = train_statistics["p50"]
            repaired.at[index, "heldout_p50"] = heldout_statistics["p50"]
            repaired.at[index, "train_p90"] = train_statistics["p90"]
            repaired.at[index, "heldout_p90"] = heldout_statistics["p90"]
        return repaired

    setattr(domain_shift_summary, _PATCH_MARKER, True)
    module.domain_shift_summary = domain_shift_summary
