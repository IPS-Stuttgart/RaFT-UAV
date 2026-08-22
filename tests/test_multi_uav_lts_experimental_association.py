from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts._proposal_common_motion import estimate_common_motion
from raft_uav.multi_uav_lts._proposal_seed_calibration import calibrate_proposals
from raft_uav.multi_uav_lts._records import Detection


@dataclasses.dataclass(frozen=True)
class _Node:
    index: int
    row: Detection


@dataclasses.dataclass(frozen=True)
class _MotionParameters:
    enable_common_motion: bool = True
    common_motion_min_pairs: int = 4
    common_motion_max_normalized_step: float = 8.0
    common_motion_max_normalized_residual: float = 0.5


def _row(
    frame: int,
    object_id: int,
    x: float,
    *,
    y: float = 10.0,
    width: float = 4.0,
    height: float = 4.0,
) -> Detection:
    return Detection(frame, object_id, x, y, width, height, 0.9, 1, 1.0)


def test_fast_common_motion_recovers_shared_translation() -> None:
    nodes: list[_Node] = []
    starts = [0.0, 50.0, 100.0, 150.0]
    for frame in (1, 2, 3):
        for object_id, start in enumerate(starts, start=1):
            nodes.append(
                _Node(
                    len(nodes),
                    _row(frame, object_id, start + 20.0 * (frame - 1)),
                )
            )

    assert estimate_common_motion(nodes, _MotionParameters()) == {
        1: (20.0, 0.0),
        2: (20.0, 0.0),
    }


def test_fast_common_motion_rejects_incoherent_displacements() -> None:
    starts = [0.0, 50.0, 100.0, 150.0]
    shifts = [-20.0, -5.0, 10.0, 30.0]
    nodes = [
        _Node(index, _row(1, index + 1, start))
        for index, start in enumerate(starts)
    ]
    nodes.extend(
        _Node(
            len(nodes) + index,
            _row(2, index + 1, start + shift),
        )
        for index, (start, shift) in enumerate(
            zip(starts, shifts, strict=True)
        )
    )

    parameters = dataclasses.replace(
        _MotionParameters(),
        common_motion_max_normalized_residual=0.25,
    )
    assert estimate_common_motion(nodes, parameters) == {}


def test_seed_calibration_recovers_center_and_size_bias() -> None:
    seeds = tuple(
        _row(1, object_id, x)
        for object_id, x in enumerate((10.0, 30.0, 50.0), start=1)
    )
    proposals = tuple(
        _row(
            frame,
            object_id,
            x + 2.0,
            y=11.0,
            width=5.0,
            height=3.0,
        )
        for frame in (1, 2)
        for object_id, x in enumerate((10.0, 30.0, 50.0), start=1)
    )

    calibrated, summary = calibrate_proposals(seeds, proposals)

    assert summary.applied
    assert summary.matched_pairs == 3
    assert summary.inlier_pairs == 3
    for expected, actual in zip(seeds, calibrated[:3], strict=True):
        assert actual.x1 == pytest.approx(expected.x1)
        assert actual.y1 == pytest.approx(expected.y1)
        assert actual.width == pytest.approx(expected.width)
        assert actual.height == pytest.approx(expected.height)


def test_seed_calibration_rejects_excessive_shift() -> None:
    seeds = (
        _row(1, 1, 0.0),
        _row(1, 2, 20.0),
    )
    proposals = (
        _row(1, 1, 100.0),
        _row(1, 2, 120.0),
    )

    calibrated, summary = calibrate_proposals(
        seeds,
        proposals,
        max_match_center_distance=30.0,
    )

    assert not summary.applied
    assert calibrated == proposals


def _write_jpeg(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        bytes([8])
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + bytes([3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0])
    )
    path.write_bytes(
        b"\xff\xd8"
        + b"\xff\xe0"
        + (4).to_bytes(2, "big")
        + b"xx"
        + b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + payload
        + b"\xff\xd9"
    )


def _evidence_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_improved_evidence",
            scripts / "run_multi_uav_lts_improved_evidence.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to load improved evidence module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_native_resolution_inventory_groups_sequences(tmp_path: Path) -> None:
    module = _evidence_module()
    image_root = tmp_path / "images"
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    for sequence, width in (("A", 512), ("B", 640), ("C", 512)):
        (seed_dir / f"{sequence}.txt").write_text("seed\n", encoding="utf-8")
        _write_jpeg(image_root / sequence / "0001.jpg", width, 512)
        _write_jpeg(image_root / sequence / "0002.jpg", width, 512)

    assert module._sequence_resolution_groups(image_root, seed_dir) == (
        ((512, 512), ("A", "C")),
        ((640, 512), ("B",)),
    )


def test_resume_prefix_discards_uncertain_tail(tmp_path: Path) -> None:
    module = _evidence_module()
    image_root = tmp_path / "images"
    baseline_root = tmp_path / "baseline"
    for sequence in ("A", "B", "C", "D"):
        (image_root / sequence).mkdir(parents=True)
    for sequence in ("A", "B", "C"):
        for directory in ("proposals", "predictions"):
            path = baseline_root / directory / f"{sequence}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("row\n", encoding="utf-8")

    retained, removed = module._retain_safe_baseline_prefix(
        baseline_root,
        image_root,
    )

    assert retained == 2
    assert removed == 2
    assert sorted(
        path.stem for path in (baseline_root / "predictions").glob("*.txt")
    ) == ["A", "B"]
