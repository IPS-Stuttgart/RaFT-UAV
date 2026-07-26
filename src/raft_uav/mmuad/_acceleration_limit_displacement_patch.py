"""Report Track 5 acceleration-repair displacement from input to final output."""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_reports_net_acceleration_displacement"
_COORDINATE_COLUMNS = ("state_x_m", "state_y_m", "state_z_m")


def install() -> None:
    """Install net-displacement diagnostics on acceleration-kink repair."""

    from raft_uav.mmuad import track5_acceleration_limit as acceleration_limit

    original = acceleration_limit._repair_sequence
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _repair_sequence(group, **kwargs):
        original_positions = pd.DataFrame(group)[list(_COORDINATE_COLUMNS)].to_numpy(
            float,
        )
        repaired, diagnostics = original(group, **kwargs)
        final_positions = repaired[list(_COORDINATE_COLUMNS)].to_numpy(float)
        net_displacement = np.linalg.norm(
            final_positions - original_positions,
            axis=1,
        )
        moved = np.isfinite(net_displacement) & (net_displacement > 1.0e-9)

        repaired = repaired.copy()
        diagnostics = diagnostics.copy()
        for rows in (repaired, diagnostics):
            applied = rows["acceleration_limit_applied"].to_numpy(bool) & moved
            rows["acceleration_limit_applied"] = applied
            rows["acceleration_limit_displacement_m"] = np.where(
                applied,
                net_displacement,
                0.0,
            )
            rows["acceleration_limit_iteration"] = np.where(
                applied,
                rows["acceleration_limit_iteration"].to_numpy(int),
                0,
            )
        return repaired, diagnostics

    setattr(_repair_sequence, _PATCH_MARKER, True)
    acceleration_limit._repair_sequence = _repair_sequence
    implementation = getattr(acceleration_limit, "_IMPL", None)
    if implementation is not None:
        implementation._repair_sequence = _repair_sequence
