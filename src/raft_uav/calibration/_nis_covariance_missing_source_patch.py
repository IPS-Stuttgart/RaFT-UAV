"""Drop diagnostics rows with missing sources before NIS calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.calibration import nis_covariance as _IMPL


def _normalized_diagnostics_frame(
    frame: pd.DataFrame,
    *,
    accepted_only: bool,
) -> pd.DataFrame:
    """Normalize valid diagnostics without turning null sources into labels."""

    required = {"source", "measurement_dim", "nis"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"diagnostics frame is missing required columns: {missing}")

    work = frame.copy()
    if accepted_only and "accepted" in work.columns:
        work = work.loc[work["accepted"].map(_IMPL._truthy)].copy()
    work["measurement_dim"] = pd.to_numeric(work["measurement_dim"], errors="coerce")
    work["nis"] = pd.to_numeric(work["nis"], errors="coerce")
    work = work.dropna(subset=["source", "measurement_dim", "nis"])
    work["source"] = work["source"].astype(str)
    work = work.loc[np.isfinite(work["nis"].to_numpy(dtype=float))]
    work = work.loc[work["nis"].to_numpy(dtype=float) >= 0.0]
    work["measurement_dim"] = work["measurement_dim"].astype(int)
    work = work.loc[work["measurement_dim"] > 0]
    return work


_IMPL._normalized_diagnostics_frame = _normalized_diagnostics_frame
