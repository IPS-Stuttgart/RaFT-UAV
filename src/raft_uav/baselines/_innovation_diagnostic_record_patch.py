"""Keep non-finite NIS values from masquerading as zero distance."""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import numpy as np

_PATCH_MARKER = "_raft_uav_handles_nonfinite_nis_distance"


def apply_innovation_diagnostic_record_patch(module: ModuleType) -> None:
    """Patch compact diagnostic export to keep invalid distances unknown."""

    original = module.raft_innovation_diagnostic_record
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def raft_innovation_diagnostic_record(diagnostic):
        record = original(diagnostic)
        nis = diagnostic.nis
        if nis is not None and not np.isfinite(float(nis)):
            record["mahalanobis_distance"] = None
        return record

    setattr(raft_innovation_diagnostic_record, _PATCH_MARKER, True)
    module.raft_innovation_diagnostic_record = raft_innovation_diagnostic_record
