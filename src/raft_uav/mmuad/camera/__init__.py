"""Compatibility wrapper for specificity-aware camera calibration lookup.

The maintained implementation lives in the sibling ``camera.py`` module. This
package preserves the public import path while selecting the most specific
one-way source-prefix match when multiple camera calibrations are available.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_IMPL_PATH = Path(__file__).resolve().parent.parent / "camera.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._camera_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load camera implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _model_for_source(models, source):
    """Return the exact or longest one-way source-prefix camera model."""

    source_key = str(source).strip().lower()
    normalized = [
        (str(key).strip().lower(), model)
        for key, model in models.items()
    ]
    for key, model in normalized:
        if source_key == key:
            return model
    if len(models) == 1:
        return next(iter(models.values()))
    matches = [
        (len(key), model)
        for key, model in normalized
        if key and source_key.startswith(key)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


_IMPL._model_for_source = _model_for_source

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_model_for_source"] = _model_for_source

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
