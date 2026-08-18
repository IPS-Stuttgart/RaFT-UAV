from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from raft_uav.multi_uav_lts._records import (
    Detection,
    format_detection,
    parse_detection_text,
)
from raft_uav.multi_uav_lts.hota_tube import apply_hota_tube


def row(
    frame: int,
    object_id: int,
    x: float,
    *,
    y: float = 0.0,
    width: float = 10.0,
    height: float = 10.0,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame,
        object_id,
        x,
        y,
        width,
        height,
        confidence,
        1,
        1.0,
    )


def write_rows(path: Path, rows: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(format_detection(item) + "\n" for item in rows),
        encoding="utf-8",
    )


def read_rows(path: Path) -> list[Detection]:
    return parse_detection_text(path.read_text(encoding="utf-8"), source=str(path))


def test_inserts_and_expands_only_the_synthetic_gap_row(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    observed = [row(1, 7, 0.0), row(3, 7, 4.0)]
    write_rows(source / "SEQ.txt", observed)

    summary = apply_hota_tube(source, output)
    result = read_rows(output / "SEQ.txt")

    assert result[0] == observed[0]
    assert result[2] == observed[1]
    assert [item.frame_id for item in result] == [1, 2, 3]
    assert result[1].object_id == 7
    assert result[1].center_x == pytest.approx(7.0)
    assert result[1].width == pytest.approx(17.5)
    assert result[1].height == pytest.approx(17.5)
    assert result[1].confidence == pytest.approx(0.81)
    assert summary.inserted_rows == 1
    assert summary.maximum_scale == pytest.approx(1.75)


def test_velocity_disagreement_increases_the_tube_width(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_rows(
        source / "SEQ.txt",
        [
            row(1, 1, 0.0),
            row(2, 1, 1.0),
            row(4, 1, 20.0),
            row(5, 1, 40.0),
        ],
    )

    summary = apply_hota_tube(
        source,
        output,
        base_inflation=0.0,
        velocity_inflation=1.0,
        max_scale=10.0,
    )

    synthetic = next(
        item for item in read_rows(output / "SEQ.txt") if item.frame_id == 3
    )
    assert synthetic.width > 10.0
    assert summary.maximum_scale > 1.0


def test_conflict_gate_rejects_an_ambiguous_synthetic_box(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_rows(
        source / "SEQ.txt",
        [
            row(1, 1, 0.0),
            row(3, 1, 4.0),
            row(2, 2, 0.0, width=20.0, height=20.0),
        ],
    )

    summary = apply_hota_tube(source, output, conflict_iou=0.1)
    result = read_rows(output / "SEQ.txt")

    assert [(item.frame_id, item.object_id) for item in result] == [
        (1, 1),
        (2, 2),
        (3, 1),
    ]
    assert summary.inserted_rows == 0
    assert summary.conflict_rejected_rows == 1


def test_zero_conflict_threshold_keeps_a_disjoint_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_rows(
        source / "SEQ.txt",
        [
            row(1, 1, 0.0),
            row(3, 1, 4.0),
            row(2, 2, 100.0),
        ],
    )

    summary = apply_hota_tube(source, output, conflict_iou=0.0)

    assert summary.inserted_rows == 1
    assert {
        (item.frame_id, item.object_id)
        for item in read_rows(output / "SEQ.txt")
    } == {
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 1),
    }


def test_does_not_fill_a_gap_beyond_the_selected_horizon(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    observed = [row(1, 1, 0.0), row(4, 1, 6.0)]
    write_rows(source / "SEQ.txt", observed)

    summary = apply_hota_tube(source, output, max_gap=1)

    assert read_rows(output / "SEQ.txt") == observed
    assert summary.eligible_gaps == 0
    assert summary.inserted_rows == 0


def test_preserves_the_mot_unknown_confidence_sentinel(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_rows(
        source / "SEQ.txt",
        [
            row(1, 1, 0.0, confidence=-1.0),
            row(3, 1, 4.0, confidence=-1.0),
        ],
    )

    apply_hota_tube(source, output)
    synthetic = next(
        item for item in read_rows(output / "SEQ.txt") if item.frame_id == 2
    )

    assert synthetic.confidence == -1.0


def test_reads_a_root_level_prediction_zip(tmp_path: Path) -> None:
    output = tmp_path / "output"
    archive_path = tmp_path / "predictions.zip"
    payload = "".join(
        format_detection(item) + "\n"
        for item in (row(1, 3, 0.0), row(3, 3, 4.0))
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("SEQ.txt", payload)

    summary = apply_hota_tube(archive_path, output)

    assert summary.sequence_count == 1
    assert len(read_rows(output / "SEQ.txt")) == 3


def test_subset_mode_rejects_unknown_sequences(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_rows(source / "SEQ.txt", [row(1, 1, 0.0), row(2, 1, 1.0)])

    with pytest.raises(ValueError, match="unknown prediction sequences"):
        apply_hota_tube(
            source,
            tmp_path / "output",
            sequences=("MISSING",),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_gap": -1}, "max_gap"),
        ({"max_gap": True}, "max_gap"),
        ({"min_track_observations": 1}, "at least 2"),
        ({"max_scale": 0.9}, "max_scale"),
        ({"conflict_iou": 1.1}, "conflict_iou"),
        ({"confidence_decay": float("nan")}, "confidence_decay"),
    ],
)
def test_rejects_invalid_parameters(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "source"
    write_rows(source / "SEQ.txt", [row(1, 1, 0.0), row(2, 1, 1.0)])

    with pytest.raises(ValueError, match=message):
        apply_hota_tube(source, tmp_path / "output", **kwargs)


def test_rejects_an_output_nested_in_the_prediction_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_rows(source / "SEQ.txt", [row(1, 1, 0.0), row(2, 1, 1.0)])

    with pytest.raises(ValueError, match="nested in predictions"):
        apply_hota_tube(source, source / "output")
