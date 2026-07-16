"""Compatibility wrapper for strict normalized Track 5 class validation.

The maintained implementation lives in the sibling
``track5_submission_ensemble.py`` module. This package preserves the public
import path while preventing malformed normalized classification values from
being silently dropped or truncated to integers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.mmuad.class_probability_context import OFFICIAL_CLASS_LABELS

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_submission_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_submission_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 submission-ensemble implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_INTERNAL_SUBMISSION_ROWS = (
    _IMPL._normalize_internal_submission_rows
)


def _normalize_internal_submission_rows(
    rows: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    """Reject invalid classes on normalized rows with usable measurements."""

    frame = pd.DataFrame(rows).copy()
    lookup = _IMPL._normalized_column_lookup(frame)
    classification_column = _IMPL._normalized_classification_column(lookup)
    measurement_columns = ("time_s", "state_x_m", "state_y_m", "state_z_m")
    if classification_column is not None and all(
        column in lookup for column in measurement_columns
    ):
        measurements = pd.DataFrame(
            {
                column: pd.to_numeric(frame[lookup[column]], errors="coerce")
                for column in measurement_columns
            },
            index=frame.index,
        )
        finite_measurements = pd.Series(
            np.isfinite(measurements.to_numpy(dtype=float)).all(axis=1),
            index=frame.index,
        )
        if bool(finite_measurements.any()):
            raw_classes = frame.loc[finite_measurements, classification_column]
            normalized_classes = _IMPL._predicted_class_labels(raw_classes)
            valid_classes = normalized_classes.isin(OFFICIAL_CLASS_LABELS)
            if not bool(valid_classes.all()):
                examples = sorted(
                    {repr(value) for value in raw_classes.loc[~valid_classes].tolist()}
                )
                raise ValueError(
                    "invalid normalized Track 5 Classification values in "
                    f"{source_path}: {', '.join(examples)}"
                )

    return _ORIGINAL_NORMALIZE_INTERNAL_SUBMISSION_ROWS(
        frame,
        source_path=source_path,
    )


_IMPL._normalize_internal_submission_rows = _normalize_internal_submission_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_internal_submission_rows"] = (
    _normalize_internal_submission_rows
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
