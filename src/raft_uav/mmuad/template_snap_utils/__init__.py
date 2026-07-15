"""Compatibility wrapper requiring exact finite Track 5 template-snap class IDs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "template_snap_utils.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._template_snap_utils_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load template-snap utilities from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _integer_classification_values(values: pd.Series) -> pd.Series:
    """Return exact finite integer-valued official classification cells."""

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
    nonfinite = numbers.notna().to_numpy() & ~finite
    if nonfinite.any():
        row_index = int(np.flatnonzero(nonfinite)[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be finite integer ids; "
            f"got {bad_value!r}"
        )

    integer_like = finite & (numeric == np.rint(numeric))
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
            list(_IMPL.OFFICIAL_TRACK5_CLASS_IDS),
        )
    if invalid_domain.any():
        row_index = int(np.flatnonzero(invalid_domain)[0])
        class_id = int(np.rint(numeric[row_index]))
        allowed = ", ".join(
            str(item) for item in sorted(_IMPL.OFFICIAL_TRACK5_CLASS_IDS)
        )
        raise ValueError(
            "official MMUAD Classification values must be one of "
            f"{{{allowed}}}; got {class_id!r}"
        )
    return numbers


_IMPL._integer_classification_values = _integer_classification_values

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_integer_classification_values"] = _integer_classification_values

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
