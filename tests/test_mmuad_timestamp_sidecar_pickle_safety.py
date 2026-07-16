from __future__ import annotations

from pathlib import Path

import numpy as np

from raft_uav.mmuad.sequence import (
    _ordered_timestamps_from_timestamp_sidecar,
    _timestamp_sidecar_explicit_map,
)


def _write_marker(path: str) -> float:
    Path(path).write_text("executed", encoding="utf-8")
    return 12.5


class _PicklePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def test_numpy_timestamp_array_does_not_deserialize_pickled_objects(tmp_path: Path) -> None:
    marker = tmp_path / "ordered-payload-executed"
    sidecar = tmp_path / "timestamps.npy"
    np.save(sidecar, np.array([_PicklePayload(marker)], dtype=object))

    assert _ordered_timestamps_from_timestamp_sidecar(sidecar) == []
    assert not marker.exists()


def test_numpy_timestamp_map_does_not_deserialize_pickled_objects(tmp_path: Path) -> None:
    marker = tmp_path / "mapping-payload-executed"
    sidecar = tmp_path / "frame_timestamps.npz"
    np.savez(
        sidecar,
        filename=np.array([_PicklePayload(marker)], dtype=object),
        time_s=np.array([12.5]),
    )

    assert _timestamp_sidecar_explicit_map(sidecar) == {}
    assert not marker.exists()


def test_non_object_numpy_timestamp_sidecars_remain_supported(tmp_path: Path) -> None:
    ordered_sidecar = tmp_path / "timestamps.npy"
    np.save(ordered_sidecar, np.array([1.25, 2.5]))

    mapping_sidecar = tmp_path / "frame_timestamps.npz"
    np.savez(
        mapping_sidecar,
        filename=np.array(["frame-1.pcd", "frame-2.pcd"]),
        time_s=np.array([1.25, 2.5]),
    )

    assert _ordered_timestamps_from_timestamp_sidecar(ordered_sidecar) == [1.25, 2.5]
    timestamp_map = _timestamp_sidecar_explicit_map(mapping_sidecar)
    assert timestamp_map["frame-1.pcd"] == 1.25
    assert timestamp_map["frame-2.pcd"] == 2.5
