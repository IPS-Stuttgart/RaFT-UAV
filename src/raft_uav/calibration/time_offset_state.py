"""Online time-offset state helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyrecest.filters.online_time_offset_estimator import (
    OnlineTimeOffsetEstimator as _PyRecEstOnlineTimeOffsetEstimator,
)


@dataclass
class OnlineTimeOffsetEstimator:
    """RaFT-UAV compatibility wrapper around PyRecEst's offset estimator."""

    offset_s: float = 0.0
    variance_s2: float = 1.0
    process_variance_s2: float = 1.0e-4
    min_speed_mps: float = 1.0
    _estimator: _PyRecEstOnlineTimeOffsetEstimator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = _PyRecEstOnlineTimeOffsetEstimator(
            offset=self.offset_s,
            variance=self.variance_s2,
            process_variance=self.process_variance_s2,
            min_speed=self.min_speed_mps,
        )
        self._sync_from_pyrecest()

    def predict(self, dt_s: float = 0.0) -> None:
        """Apply random-walk process noise."""

        self._sync_to_pyrecest()
        self._estimator.predict(dt=dt_s)
        self._sync_from_pyrecest()

    def update_from_position_residual(
        self,
        *,
        residual_m: np.ndarray,
        velocity_mps: np.ndarray,
        measurement_variance_m2: float,
    ) -> float:
        """Update the offset from a position residual and return innovation NIS."""

        self._sync_to_pyrecest()
        nis = self._estimator.update_from_position_residual(
            residual=residual_m,
            velocity=velocity_mps,
            measurement_variance=measurement_variance_m2,
        )
        self._sync_from_pyrecest()
        return nis

    @property
    def std_s(self) -> float:
        """Return the posterior offset standard deviation."""

        self._sync_to_pyrecest()
        return self._estimator.std

    def _sync_to_pyrecest(self) -> None:
        self._estimator.offset = float(self.offset_s)
        self._estimator.variance = float(self.variance_s2)
        self._estimator.process_variance = float(self.process_variance_s2)
        self._estimator.min_speed = float(self.min_speed_mps)

    def _sync_from_pyrecest(self) -> None:
        self.offset_s = float(self._estimator.offset)
        self.variance_s2 = float(self._estimator.variance)
        self.process_variance_s2 = float(self._estimator.process_variance)
        self.min_speed_mps = float(self._estimator.min_speed)


def apply_time_offset(frame: pd.DataFrame, *, offset_s: float, column: str = "time_s") -> pd.DataFrame:
    """Return a copy of ``frame`` with ``column`` shifted by ``offset_s``."""

    out = frame.copy()
    if column not in out.columns:
        raise KeyError(column)
    out[column] = pd.to_numeric(out[column], errors="coerce") + float(offset_s)
    return out
