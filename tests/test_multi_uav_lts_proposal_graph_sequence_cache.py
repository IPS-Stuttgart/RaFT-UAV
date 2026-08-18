from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts import experimental_proposal_graph_tracker
from raft_uav.multi_uav_lts import proposal_graph_tracker
from raft_uav.multi_uav_lts._records import Detection, format_detection


def _row(frame: int, object_id: int, x: float) -> Detection:
    return Detection(frame, object_id, x, 10.0, 6.0, 6.0, 0.9, 1, 1.0)


def _write_rows(path: Path, rows: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(format_detection(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    proposal_dir = tmp_path / "proposal-source" / "proposals"
    seed_dir = tmp_path / "seeds"
    for sequence, offset in (("A", 0.0), ("B", 30.0)):
        _write_rows(seed_dir / f"{sequence}.txt", [_row(1, 1, offset)])
        _write_rows(
            proposal_dir / f"{sequence}.txt",
            [
                _row(1, 1, offset),
                _row(2, 1, offset + 1.0),
                _row(3, 1, offset + 2.0),
            ],
        )
    return proposal_dir, seed_dir


def _arguments(
    proposal_dir: Path,
    seed_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        str(proposal_dir),
        "--first-frame-label-dir",
        str(seed_dir),
        "--output-dir",
        str(output_dir),
        "--sequences",
        "A",
        "B",
    ]


def test_sequence_cache_reuses_complete_outputs_across_run_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal_dir, seed_dir = _inputs(tmp_path)
    real_tracker = proposal_graph_tracker.track_proposal_graph
    calls: list[str] = []

    def counted_tracker(proposals, seeds, output, **kwargs):
        labels = sorted(Path(seeds).glob("*.txt"))
        assert len(labels) == 1
        calls.append(labels[0].stem)
        return real_tracker(proposals, seeds, output, **kwargs)

    monkeypatch.setattr(
        proposal_graph_tracker,
        "track_proposal_graph",
        counted_tracker,
    )

    first_output = tmp_path / "run-one" / "predictions"
    second_output = tmp_path / "run-two" / "predictions"
    assert (
        experimental_proposal_graph_tracker.main(
            _arguments(proposal_dir, seed_dir, first_output)
        )
        == 0
    )
    assert calls == ["A", "B"]

    assert (
        experimental_proposal_graph_tracker.main(
            _arguments(proposal_dir, seed_dir, second_output)
        )
        == 0
    )
    assert calls == ["A", "B"]
    for sequence in ("A", "B"):
        assert (second_output / f"{sequence}.txt").read_bytes() == (
            first_output / f"{sequence}.txt"
        ).read_bytes()


def test_sequence_cache_invalidates_only_changed_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal_dir, seed_dir = _inputs(tmp_path)
    real_tracker = proposal_graph_tracker.track_proposal_graph
    calls: list[str] = []

    def counted_tracker(proposals, seeds, output, **kwargs):
        labels = sorted(Path(seeds).glob("*.txt"))
        assert len(labels) == 1
        calls.append(labels[0].stem)
        return real_tracker(proposals, seeds, output, **kwargs)

    monkeypatch.setattr(
        proposal_graph_tracker,
        "track_proposal_graph",
        counted_tracker,
    )

    assert (
        experimental_proposal_graph_tracker.main(
            _arguments(
                proposal_dir,
                seed_dir,
                tmp_path / "run-one" / "predictions",
            )
        )
        == 0
    )
    assert calls == ["A", "B"]

    _write_rows(
        proposal_dir / "B.txt",
        [
            _row(1, 1, 30.0),
            _row(2, 1, 32.0),
            _row(3, 1, 34.0),
        ],
    )
    assert (
        experimental_proposal_graph_tracker.main(
            _arguments(
                proposal_dir,
                seed_dir,
                tmp_path / "run-two" / "predictions",
            )
        )
        == 0
    )
    assert calls == ["A", "B", "B"]


def test_no_sequence_cache_preserves_single_process_behavior(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal_dir, seed_dir = _inputs(tmp_path)
    real_tracker = proposal_graph_tracker.track_proposal_graph
    calls: list[tuple[str, ...]] = []

    def counted_tracker(proposals, seeds, output, **kwargs):
        calls.append(tuple(kwargs.get("sequences") or ()))
        return real_tracker(proposals, seeds, output, **kwargs)

    monkeypatch.setattr(
        proposal_graph_tracker,
        "track_proposal_graph",
        counted_tracker,
    )
    arguments = [
        "--no-sequence-cache",
        *_arguments(
            proposal_dir,
            seed_dir,
            tmp_path / "uncached" / "predictions",
        ),
    ]

    assert experimental_proposal_graph_tracker.main(arguments) == 0
    assert calls == [("A", "B")]