"""Strict Track 5 template snapping utilities.

This package wrapper loads the legacy module implementation from the sibling
``template_snap_utils.py`` file and tightens the official Classification parser
before re-exporting the public helpers.  The wrapper is intentionally small so
existing imports such as ``raft_uav.mmuad.template_snap_utils`` keep working
while near-integer class labels are rejected instead of rounded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import OFFICIAL_TRACK5_CLASS_IDS

_IMPL_PATH = Path(__file__).resolve().parent.parent / "template_snap_utils.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._template_snap_utils_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy template snap utilities from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)


def _integer_classification_values(values: pd.Series) -> pd.Series:
    raw = pd.Series(values)
    boolean_mask = raw.map(lambda value: isinstance(value, (bool, np.bool_)))
    if boolean_mask.any():
        row_index = int(np.flatnonzero(boolean_mask.to_numpy())[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids, not booleans; "
            f"got {bad_value!r}"
        )
    numbers = pd.to_numeric(raw, errors="coerce")
    bad_text_mask = numbers.isna() & raw.notna()
    if bad_text_mask.any():
        row_index = int(np.flatnonzero(bad_text_mask.to_numpy())[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids; "
            f"got {bad_value!r}"
        )
    numeric = numbers.to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    integer_like = finite & np.isclose(
        numeric,
        np.rint(numeric),
        rtol=0.0,
        atol=1.0e-12,
    )
    fractional = finite & ~integer_like
    if fractional.any():
        row_index = int(np.flatnonzero(fractional)[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids; "
            f"got {bad_value!r}"
        )
    invalid_domain = np.zeros_like(finite, dtype=bool)
    if integer_like.any():
        integer_values = np.rint(numeric[integer_like]).astype(int)
        invalid_domain[integer_like] = ~np.isin(
            integer_values,
            list(OFFICIAL_TRACK5_CLASS_IDS),
        )
    if invalid_domain.any():
        row_index = int(np.flatnonzero(invalid_domain)[0])
        class_id = int(np.rint(numeric[row_index]))
        allowed = ", ".join(str(item) for item in sorted(OFFICIAL_TRACK5_CLASS_IDS))
        raise ValueError(
            "official MMUAD Classification values must be one of "
            f"{{{allowed}}}; got {class_id!r}"
        )
    return numbers


_IMPL._integer_classification_values = _integer_classification_values

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

globals()["_integer_classification_values"] = _integer_classification_values
__doc__ = _IMPL.__doc__
__all__ = [_name for _name in globals() if not _name.startswith("__")]
