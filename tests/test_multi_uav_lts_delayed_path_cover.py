from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts._records import (
    Detection,
    format_detection,
    parse_detection_text,
)
from raft_uav.multi_uav_lts.experimental_proposal_graph_tracker import (
    main as experimental_main,
)
from raft_uav.multi_uav_lts.proposal_graph_tracker import track_proposal_graph


def row(frame: int, proposal_id: int, x: float) -> Detection:
    return Detection(frame, proposal_id, x, 10.0, 4.0, 4.0, 0.9, 1, 1.0)


def write_rows(path: Path, rows: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(format_detection(item) + "\n" for item in rows),
        encoding="utf-8",
    )


def output_x_by_id(path: Path) -> dict[int, list[float]]:
    rows = parse_detection_text(path.read_text(encoding="utf-8"), source=str(path))
    result: dict[int, list[float]] = {}
    for item in rows:
        result.setdefault(item.object_id, []).append(item.x1)
    return result


def _crossing_inputs(tmp_path: Path) -> tuple[Path, Path]:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    seeds = [row(1, 1, 0.0), row(1, 2, 10.0)]
    proposal_rows = [
        row(frame, frame * 10 + index, x)
        for frame, positions in enumerate(
            ((0.0, 10.0), (4.0, 6.0), (8.0, 2.0), (12.0, -2.0)),
            start=1,
        )
        for index, x in enumerate(positions, start=1)
    ]
    write_rows(labels / "SEQ.txt", seeds)
    write_rows(proposals / "SEQ.txt", proposal_rows)
    return labels, proposals


def test_delayed_path_cover_uses_future_evidence_to_resolve_crossing(
    tmp_path: Path,
) -> None:
    labels, proposals = _crossing_inputs(tmp_path)
    raw_output = tmp_path / "raw"
    delayed_output = tmp_path / "delayed"

    track_proposal_graph(
        proposals,
        labels,
        raw_output,
        max_link_gap=0,
        birth_min_hits=100,
    )
    assert output_x_by_id(raw_output / "SEQ.txt") == {
        1: [0.0, 4.0, 2.0, -2.0],
        2: [10.0, 6.0, 8.0, 12.0],
    }

    assert (
        experimental_main(
            [
                "--no-sequence-cache",
                str(proposals),
                "--first-frame-label-dir",
                str(labels),
                "--output-dir",
                str(delayed_output),
                "--max-link-gap",
                "0",
                "--birth-min-hits",
                "100",
                "--enable-delayed-path-cover",
                "--delayed-max-gap",
                "0",
                "--delayed-lookahead-frames",
                "2",
                "--delayed-successors-per-frame",
                "2",
                "--delayed-continuation-weight",
                "0.75",
            ]
        )
        == 0
    )
    assert output_x_by_id(delayed_output / "SEQ.txt") == {
        1: [0.0, 4.0, 8.0, 12.0],
        2: [10.0, 6.0, 2.0, -2.0],
    }


def test_ambiguity_beam_uses_second_order_motion_to_resolve_crossing(
    tmp_path: Path,
) -> None:
    labels, proposals = _crossing_inputs(tmp_path)
    pairwise_output = tmp_path / "pairwise"
    beam_output = tmp_path / "beam"
    common = [
        "--no-sequence-cache",
        str(proposals),
        "--first-frame-label-dir",
        str(labels),
        "--max-link-gap",
        "0",
        "--birth-min-hits",
        "100",
        "--enable-delayed-path-cover",
        "--delayed-max-gap",
        "0",
        "--delayed-lookahead-frames",
        "2",
        "--delayed-successors-per-frame",
        "2",
        "--delayed-continuation-weight",
        "0",
    ]

    assert experimental_main([*common, "--output-dir", str(pairwise_output)]) == 0
    assert output_x_by_id(pairwise_output / "SEQ.txt") == {
        1: [0.0, 4.0, 2.0, -2.0],
        2: [10.0, 6.0, 8.0, 12.0],
    }

    assert (
        experimental_main(
            [
                *common,
                "--output-dir",
                str(beam_output),
                "--enable-ambiguity-beam",
                "--ambiguity-beam-width",
                "8",
                "--ambiguity-beam-margin",
                "2",
                "--ambiguity-acceleration-weight",
                "1",
            ]
        )
        == 0
    )
    assert output_x_by_id(beam_output / "SEQ.txt") == {
        1: [0.0, 4.0, 8.0, 12.0],
        2: [10.0, 6.0, 2.0, -2.0],
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--delayed-max-gap", "-1"], "max_gap"),
        (["--delayed-lookahead-frames", "0"], "lookahead_frames"),
        (["--delayed-successors-per-frame", "0"], "successors_per_frame"),
        (["--delayed-continuation-weight", "nan"], "continuation_weight"),
        (["--delayed-continuation-clip", "0"], "continuation_clip"),
    ],
)
def test_delayed_path_cover_rejects_invalid_controls(
    arguments: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        experimental_main(["unused", "--enable-delayed-path-cover", *arguments])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--ambiguity-beam-width", "0"], "width"),
        (["--ambiguity-beam-max-component-nodes", "0"], "max_component_nodes"),
        (["--ambiguity-beam-margin", "nan"], "margin"),
        (["--ambiguity-acceleration-weight", "-1"], "acceleration_weight"),
        (["--ambiguity-acceleration-clip", "0"], "acceleration_clip"),
        (["--ambiguity-beam-expansion-factor", "0"], "expansion_factor"),
    ],
)
def test_ambiguity_beam_rejects_invalid_controls(
    arguments: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        experimental_main(["unused", "--enable-delayed-path-cover", *arguments])


def test_ambiguity_beam_requires_delayed_path_cover() -> None:
    with pytest.raises(ValueError, match="requires delayed path cover"):
        experimental_main(["unused", "--enable-ambiguity-beam"])
