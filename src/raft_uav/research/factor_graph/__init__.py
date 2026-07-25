"""Compatibility fixes for the factor-graph research utilities.

The maintained implementation lives in the sibling ``factor_graph.py`` module.
This package preserves the public import path while parsing row-level measurement
uncertainties defensively.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "factor_graph.py"
_LEGACY_NAME = f"{__name__.rsplit('.', 1)[0]}._factor_graph_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load factor-graph implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)


def _row_position_std(row: pd.Series, cfg: object) -> np.ndarray:
    """Return valid row uncertainty or the configured source-specific default."""

    source = str(row.get("source", "radar")).strip().casefold()
    default = float(cfg.rf_std_m if source == "rf" else cfg.measurement_std_m)

    std_columns = ("std_east_m", "std_north_m", "std_up_m")
    if all(column in row.index for column in std_columns):
        standard_deviations = [optional_float(row[column]) for column in std_columns]
        if all(value is not None and value > 0.0 for value in standard_deviations):
            return np.asarray(standard_deviations, dtype=float)

    covariance_columns = ("cov_ee", "cov_nn", "cov_uu")
    if all(column in row.index for column in covariance_columns):
        variances = [optional_float(row[column]) for column in covariance_columns]
        if all(value is not None and value >= 0.0 for value in variances):
            return np.sqrt(np.maximum(np.asarray(variances, dtype=float), 1.0e-9))

    return np.full(3, default, dtype=float)


_LEGACY._row_position_std = _row_position_std

for _name in dir(_LEGACY):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_LEGACY, _name)
