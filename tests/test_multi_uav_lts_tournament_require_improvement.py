from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.tournament import run_guarded_tournament

_TRUTH_ROWS = "1,1,0,0,10,10,1,1,1\n2,1,1,0,10,10,1,1,1\n"
_SWITCH_ROWS = "1,5,0,0,10,10,1,1,1\n2,6,1,0,10,10,1,1,1\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    raw_rows: str,
    candidate_rows: str,
) -> tuple[Path, Path, Path]:
    truth = tmp_path / "truth"
    raw = tmp_path / "raw"
    candidate = tmp_path / "candidate"
    for sequence in ("AA_00", "BB_00"):
        _write(truth / f"{sequence}.txt", _TRUTH_ROWS)
        _write(raw / f"{sequence}.txt", raw_rows)
        _write(candidate / f"{sequence}.txt", candidate_rows)
    return truth, raw, candidate


def test_require_improvement_does_not_publish_raw_fallback(tmp_path: Path) -> None:
    truth, raw, candidate = _fixture(
        tmp_path,
        raw_rows=_TRUTH_ROWS,
        candidate_rows=_SWITCH_ROWS,
    )
    output = tmp_path / "out"
    _write(output / "sentinel.txt", "keep\n")

    with pytest.raises(
        RuntimeError,
        match="no transformed candidate cleared the configured guards",
    ):
        run_guarded_tournament(
            raw,
            truth,
            output,
            candidates=(("candidate", candidate),),
            fold_count=2,
            expected_sequence_count=2,
            bootstrap_samples=20,
            require_improvement=True,
            copy_selected=True,
        )

    assert {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    } == {"sentinel.txt"}
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "keep\n"


def test_require_improvement_publishes_supported_candidate(tmp_path: Path) -> None:
    truth, raw, candidate = _fixture(
        tmp_path,
        raw_rows=_SWITCH_ROWS,
        candidate_rows=_TRUTH_ROWS,
    )
    output = tmp_path / "out"

    result = run_guarded_tournament(
        raw,
        truth,
        output,
        candidates=(("candidate", candidate),),
        fold_count=2,
        expected_sequence_count=2,
        bootstrap_samples=20,
        require_improvement=True,
        copy_selected=True,
    )

    assert result.selected_candidate == "candidate"
    assert (output / "tournament_summary.json").is_file()
    assert (output / "selected_candidate.txt").read_text(encoding="utf-8") == "candidate\n"
    for sequence in ("AA_00", "BB_00"):
        copied = output / "selected_predictions" / f"{sequence}.txt"
        assert copied.read_text(encoding="utf-8") == _TRUTH_ROWS
