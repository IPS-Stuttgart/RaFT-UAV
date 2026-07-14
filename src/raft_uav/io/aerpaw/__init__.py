"""Compatibility package with robust AERPAW dataset-root discovery.

The maintained implementation lives in the sibling ``aerpaw.py`` module. This
package preserves the public import path while making recursive discovery honor
both supported RF/radar directory spellings and reject matching regular files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_IMPL_PATH = Path(__file__).resolve().parent.parent / "aerpaw.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.io._aerpaw_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load AERPAW IO implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_RF_RADAR_ROOT_NAMES = ("RF Sensor and Radar", "RF_Sensor_and_Radar")


def find_rf_sensor_and_radar_root(dataset_root: Path) -> Path:
    """Find a supported RF/radar directory below an extracted dataset root."""

    root = Path(dataset_root)
    if root.is_dir() and root.name in _RF_RADAR_ROOT_NAMES:
        return root
    for name in _RF_RADAR_ROOT_NAMES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name in _RF_RADAR_ROOT_NAMES:
            return candidate
    raise FileNotFoundError(f"Could not find RF Sensor and Radar folder under {root}")


_IMPL.find_rf_sensor_and_radar_root = find_rf_sensor_and_radar_root

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["find_rf_sensor_and_radar_root"] = find_rf_sensor_and_radar_root

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
