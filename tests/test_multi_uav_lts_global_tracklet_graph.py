from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts._records import (
    Detection,
    format_detection,
    parse_detection_text,
)
from raft_uav.multi_uav_lts.global_tracklet_graph import (
    GlobalTrackletGraphParameters,
    _LinkCandidate,
    _Tracklet,
    _select_sparse_links,
    track_global_proposal_graph,
)


def _row(
    frame: int,
    object_id: int,
    x: float,
    *,
    y: float = 10.0,
    width: float = 4.0,
    height: float = 4.0,
    confidence: float = 0.9,
) -> Detection:
    return Detection(frame, object_id, x, y, width, height, confidence, 1, 1.0)


def _write_rows(path: Path, rows: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(format_detection(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[Detection]:
    return parse_detection_text(path.read_text(encoding="utf-8"), source=str(path))


def _run(
    tmp_path: Path,
    seeds: list[Detection],
    proposals: list[Detection],
    *,
    parameters: GlobalTrackletGraphParameters | None = None,
    suffix: str = "run",
):
    labels = tmp_path / f"labels-{suffix}"
    proposal_dir = tmp_path / f"proposals-{suffix}"
    output = tmp_path / f"output-{suffix}"
    _write_rows(labels / "S.txt", seeds)
    _write_rows(proposal_dir / "S.txt", proposals)
    summary = track_global_proposal_graph(
        proposal_dir,
        labels,
        output,
        parameters=parameters,
    )
    return summary, _read_rows(output / "S.txt"), output


def test_confirms_persistent_late_birth_and_rejects_single_frame_clutter(
    tmp_path: Path,
) -> None:
    seed = _row(1, 7, 0.0, confidence=1.0)
    proposals = [
        *[_row(frame, 100, float(frame - 1)) for frame in range(1, 7)],
        *[_row(frame, 200, 20.0 + frame) for frame in range(3, 7)],
        _row(4, 300, 80.0, confidence=0.95),
    ]
    parameters = GlobalTrackletGraphParameters(
        min_confidence=0.0,
        min_birth_frames=3,
        min_birth_span=2,
        min_birth_mean_confidence=0.5,
    )

    summary, rows, output = _run(
        tmp_path,
        [seed],
        proposals,
        parameters=parameters,
    )

    by_id = {object_id: [] for object_id in {row.object_id for row in rows}}
    for row in rows:
        by_id[row.object_id].append(row.frame_id)
    assert by_id == {7: [1, 2, 3, 4, 5, 6], 8: [3, 4, 5, 6]}
    assert summary.late_birth_paths == 1
    assert summary.dropped_paths >= 1
    payload = json.loads(
        (output / "global_tracklet_graph_summary.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "raft-uav-multi-uav-lts-global-tracklet-graph-v1"
    assert (output / "global_tracklet_graph_sequences.csv").is_file()
    assert (output / "global_tracklet_graph_links.csv").is_file()


def test_no_birth_ablation_retains_only_seeded_identity(tmp_path: Path) -> None:
    seed = _row(1, 11, 0.0, confidence=1.0)
    proposals = [
        *[_row(frame, 100, float(frame - 1)) for frame in range(1, 5)],
        *[_row(frame, 200, 30.0 + frame) for frame in range(2, 6)],
    ]
    parameters = GlobalTrackletGraphParameters(
        min_confidence=0.0,
        allow_births=False,
    )

    summary, rows, _output = _run(
        tmp_path,
        [seed],
        proposals,
        parameters=parameters,
    )

    assert {row.object_id for row in rows} == {11}
    assert summary.late_birth_paths == 0
    assert summary.dropped_paths >= 1


def test_global_gap_linking_preserves_crossing_identities(tmp_path: Path) -> None:
    seeds = [
        _row(1, 10, 0.0, width=2.0, height=2.0, confidence=1.0),
        _row(1, 20, 10.0, width=2.0, height=2.0, confidence=1.0),
    ]
    proposals = [
        _row(1, 101, 0.0, width=2.0, height=2.0),
        _row(1, 202, 10.0, width=2.0, height=2.0),
        _row(2, 101, 2.0, width=2.0, height=2.0),
        _row(2, 202, 8.0, width=2.0, height=2.0),
        # Frame 3 is deliberately missing at the crossing.
        _row(4, 303, 6.0, width=2.0, height=2.0),
        _row(4, 404, 4.0, width=2.0, height=2.0),
        _row(5, 303, 8.0, width=2.0, height=2.0),
        _row(5, 404, 2.0, width=2.0, height=2.0),
    ]
    parameters = GlobalTrackletGraphParameters(
        min_confidence=0.0,
        global_max_gap=3,
        min_birth_frames=3,
    )

    summary, rows, _output = _run(
        tmp_path,
        seeds,
        proposals,
        parameters=parameters,
    )

    positions = {(row.frame_id, row.object_id): row.x1 for row in rows}
    assert positions[(4, 10)] == pytest.approx(6.0)
    assert positions[(5, 10)] == pytest.approx(8.0)
    assert positions[(4, 20)] == pytest.approx(4.0)
    assert positions[(5, 20)] == pytest.approx(2.0)
    assert summary.selected_links == 2
    assert summary.late_birth_paths == 0


def test_border_reentry_toggle_changes_identity_continuity(tmp_path: Path) -> None:
    seed = _row(1, 7, 94.0, width=4.0, confidence=1.0)
    proposals = [
        _row(1, 100, 94.0),
        _row(2, 100, 96.0),
        _row(8, 200, 96.0),
        _row(9, 200, 95.0),
        _row(10, 200, 94.0),
    ]
    common = dict(
        min_confidence=0.0,
        global_max_gap=2,
        border_max_gap=10,
        border_margin_px=8.0,
        border_link_penalty=0.2,
        frame_width=100.0,
        frame_height=50.0,
        min_birth_frames=3,
        min_birth_span=2,
    )

    disabled_summary, disabled_rows, _ = _run(
        tmp_path,
        [seed],
        proposals,
        parameters=GlobalTrackletGraphParameters(
            **common,
            enable_border_reentry=False,
        ),
        suffix="disabled",
    )
    enabled_summary, enabled_rows, _ = _run(
        tmp_path,
        [seed],
        proposals,
        parameters=GlobalTrackletGraphParameters(
            **common,
            enable_border_reentry=True,
        ),
        suffix="enabled",
    )

    assert {row.object_id for row in disabled_rows} == {7, 8}
    assert {row.object_id for row in enabled_rows} == {7}
    assert disabled_summary.selected_border_links == 0
    assert enabled_summary.selected_border_links == 1


def test_results_are_invariant_to_proposal_row_order(tmp_path: Path) -> None:
    seed = _row(1, 5, 0.0, confidence=1.0)
    proposals = [
        *[_row(frame, 100, float(frame)) for frame in range(1, 6)],
        *[_row(frame, 200, 20.0 + frame) for frame in range(2, 6)],
        _row(3, 999, 3.0, confidence=0.2),
    ]
    parameters = GlobalTrackletGraphParameters(min_confidence=0.0)

    first_summary, first_rows, _ = _run(
        tmp_path,
        [seed],
        proposals,
        parameters=parameters,
        suffix="ordered",
    )
    second_summary, second_rows, _ = _run(
        tmp_path,
        [seed],
        list(reversed(proposals)),
        parameters=parameters,
        suffix="reversed",
    )

    assert [format_detection(row) for row in first_rows] == [
        format_detection(row) for row in second_rows
    ]
    assert first_summary.selected_links == second_summary.selected_links
    assert first_summary.late_birth_paths == second_summary.late_birth_paths


def test_sparse_global_matching_is_one_to_one() -> None:
    tracklets = [
        _Tracklet(0, [_row(1, 1, 0.0)], True),
        _Tracklet(1, [_row(1, 2, 10.0)], True),
        _Tracklet(2, [_row(3, 3, 5.0)], False),
    ]
    candidates = [
        _LinkCandidate(0, 2, 1, 0.5, 4.5, False),
        _LinkCandidate(1, 2, 1, 0.6, 4.4, False),
    ]

    selected = _select_sparse_links(tracklets, candidates)

    assert len(selected) == 1
    assert selected[0].source == 0
    assert selected[0].target == 2


def test_reciprocal_local_linking_rejects_an_exact_tie(tmp_path: Path) -> None:
    seeds = [_row(1, 1, 0.0, confidence=1.0), _row(1, 2, 10.0, confidence=1.0)]
    proposals = [_row(2, 100, 5.0)]

    reciprocal_summary, reciprocal_rows, _ = _run(
        tmp_path,
        seeds,
        proposals,
        parameters=GlobalTrackletGraphParameters(
            min_confidence=0.0,
            local_min_margin=0.1,
            reciprocal_local_links=True,
            allow_births=False,
        ),
        suffix="reciprocal",
    )
    greedy_summary, greedy_rows, _ = _run(
        tmp_path,
        seeds,
        proposals,
        parameters=GlobalTrackletGraphParameters(
            min_confidence=0.0,
            local_min_margin=0.1,
            reciprocal_local_links=False,
            allow_births=False,
        ),
        suffix="greedy",
    )

    assert [row.frame_id for row in reciprocal_rows] == [1, 1]
    assert any(row.frame_id == 2 for row in greedy_rows)
    assert reciprocal_summary.tracklet_count > greedy_summary.tracklet_count


def test_rejects_destructive_output_alias_and_invalid_border_configuration(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    _write_rows(labels / "S.txt", [_row(1, 1, 0.0)])
    _write_rows(proposals / "S.txt", [_row(1, 1, 0.0)])

    with pytest.raises(ValueError, match="output directory must be disjoint"):
        track_global_proposal_graph(proposals, labels, proposals)
    with pytest.raises(ValueError, match="output directory must be disjoint"):
        track_global_proposal_graph(proposals, labels, proposals / "nested")
    with pytest.raises(ValueError, match="frame_width and frame_height"):
        GlobalTrackletGraphParameters(enable_border_reentry=True)
    with pytest.raises(ValueError, match=r"min_birth_mean_confidence must be in \[0, 1\]"):
        GlobalTrackletGraphParameters(min_birth_mean_confidence=1.1)
    with pytest.raises(ValueError, match="reciprocal_local_links must be a Boolean"):
        GlobalTrackletGraphParameters(reciprocal_local_links=1)  # type: ignore[arg-type]


def test_rejects_unknown_sequence_and_out_of_domain_proposal_confidence(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    output = tmp_path / "output"
    _write_rows(labels / "S.txt", [_row(1, 1, 0.0)])
    _write_rows(proposals / "S.txt", [_row(1, 1, 0.0, confidence=-0.2)])

    with pytest.raises(ValueError, match="unknown first-frame sequences"):
        track_global_proposal_graph(
            proposals,
            labels,
            output,
            sequences=["missing"],
        )
    with pytest.raises(ValueError, match=r"proposal confidence must be in \[0, 1\]"):
        track_global_proposal_graph(proposals, labels, output)


def test_sparse_matching_maximizes_total_link_gain() -> None:
    tracklets = [
        _Tracklet(index, [_row(index + 1, index + 1, float(index))], False)
        for index in range(4)
    ]
    candidates = [
        _LinkCandidate(0, 2, 0, 0.0, 4.0, False),
        _LinkCandidate(0, 3, 0, 0.0, 3.0, False),
        _LinkCandidate(1, 2, 0, 0.0, 3.5, False),
        _LinkCandidate(1, 3, 0, 0.0, 0.1, False),
    ]

    selected = _select_sparse_links(tracklets, candidates)

    assert {(item.source, item.target) for item in selected} == {(0, 3), (1, 2)}
    assert sum(item.gain for item in selected) == pytest.approx(6.5)


def test_seed_file_row_order_does_not_change_assigned_identities(tmp_path: Path) -> None:
    seeds = [_row(1, 20, 10.0, confidence=1.0), _row(1, 10, 0.0, confidence=1.0)]
    proposals = [
        *[_row(frame, 100, float(frame - 1)) for frame in range(1, 5)],
        *[_row(frame, 200, 10.0 - float(frame - 1)) for frame in range(1, 5)],
    ]
    parameters = GlobalTrackletGraphParameters(min_confidence=0.0)

    _summary_a, rows_a, _ = _run(
        tmp_path,
        seeds,
        proposals,
        parameters=parameters,
        suffix="seed-order-a",
    )
    _summary_b, rows_b, _ = _run(
        tmp_path,
        list(reversed(seeds)),
        proposals,
        parameters=parameters,
        suffix="seed-order-b",
    )

    assert [format_detection(row) for row in rows_a] == [
        format_detection(row) for row in rows_b
    ]
