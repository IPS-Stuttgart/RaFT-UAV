"""Harden recursive discovery of the AERPAW RF/radar data root."""

from __future__ import annotations

from pathlib import Path


_RF_ROOT_NAMES = {"RF Sensor and Radar", "RF_Sensor_and_Radar"}


def _find_rf_sensor_and_radar_root(dataset_root: Path) -> Path:
    """Return the first valid RF/radar directory under ``dataset_root``.

    Both spellings used by extracted Dataset-28 archives are accepted at every
    nesting depth.  File-system entries with a matching name are deliberately
    ignored; callers require a directory because ``discover_flights`` iterates
    over its children.
    """

    root = Path(dataset_root)
    if root.is_dir() and root.name in _RF_ROOT_NAMES:
        return root

    for name in ("RF Sensor and Radar", "RF_Sensor_and_Radar"):
        candidate = root / name
        if candidate.is_dir():
            return candidate

    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name in _RF_ROOT_NAMES:
            return candidate

    raise FileNotFoundError(f"Could not find RF Sensor and Radar folder under {root}")


def install() -> None:
    """Install the hardened data-root lookup exactly once."""

    from raft_uav.io import aerpaw

    current = aerpaw.find_rf_sensor_and_radar_root
    if getattr(current, "_raft_uav_nested_rf_root_patch", False):
        return

    _find_rf_sensor_and_radar_root._raft_uav_nested_rf_root_patch = True  # type: ignore[attr-defined]
    aerpaw.find_rf_sensor_and_radar_root = _find_rf_sensor_and_radar_root
