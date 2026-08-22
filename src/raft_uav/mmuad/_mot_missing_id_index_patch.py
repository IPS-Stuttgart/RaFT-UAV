"""Keep anonymous MOT identities independent of DataFrame index labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

_INSTALLED = False


def _collision_safe_missing_ids(
    values: pd.Series,
    present: np.ndarray,
    *,
    prefix: str,
) -> pd.Series:
    """Replace missing IDs with deterministic row-position identities."""

    identifiers = values.astype(object).to_numpy(copy=True)
    used = {str(value) for value in identifiers[present]}
    for position in np.flatnonzero(~present):
        stem = f"__raft_uav_missing_{prefix}_{int(position)}"
        candidate = stem
        suffix = 1
        while candidate in used:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        identifiers[position] = candidate
        used.add(candidate)
    return pd.Series(
        identifiers,
        index=values.index,
        name=values.name,
        dtype=object,
    )


def install() -> None:
    """Install row-position fallback IDs at every active MOT boundary."""

    global _INSTALLED
    if _INSTALLED:
        return

    import raft_uav.mmuad as mmuad
    from raft_uav.mmuad import mot

    implementation = getattr(mot, "_IMPL", mot)
    present_mask = implementation._track_id_present_mask

    def normalized_metric_id_series(values: pd.Series, *, prefix: str) -> pd.Series:
        series = pd.Series(values, copy=True)
        present = present_mask(series).to_numpy(dtype=bool)
        return _collision_safe_missing_ids(series, present, prefix=prefix)

    def estimate_track_ids_for_metrics(estimates: pd.DataFrame) -> pd.Series:
        if "output_track_id" in estimates.columns:
            return normalized_metric_id_series(
                estimates["output_track_id"],
                prefix="estimate",
            )
        if "track_id" in estimates.columns:
            return normalized_metric_id_series(
                estimates["track_id"],
                prefix="estimate",
            )
        missing = pd.Series(None, index=estimates.index, dtype=object)
        return normalized_metric_id_series(missing, prefix="estimate")

    targets = (mot,) if implementation is mot else (mot, implementation)
    for target in targets:
        target._normalized_metric_id_series = normalized_metric_id_series
        target._estimate_track_ids_for_metrics = estimate_track_ids_for_metrics

    from raft_uav.mmuad._mot_physical_scope_patch import install as install_physical_scope

    install_physical_scope(mot=mot, mmuad=mmuad)
    _INSTALLED = True
