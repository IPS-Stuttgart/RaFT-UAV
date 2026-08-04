from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.tournament import run_guarded_tournament

_TRUTH_ROWS = "1,1,0,0,10,10,1,1,1\n2,1,1,0,10,10,1,1,1\n"
_SWITCH_ROWS = "1,5,0,0,10,10,1,1,1\n2,6,1,0,10,10,1,1,1\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _two_sequence_fixture(tmp_path: Path) -> tuple[Path, Path]:
    truth = tmp_path / "truth"
    raw = tmp_path / "raw"
    for sequence in ("AA_00", "BB_00"):
        _write(truth / f"{sequence}.txt", _TRUTH_ROWS)
        _write(raw / f"{sequence}.txt", _SWITCH_ROWS)
    return truth, raw


def test_guarded_tournament_selects_supported_improvement(tmp_path: Path) -> None:
    truth, raw = _two_sequence_fixture(tmp_path)
    candidate = tmp_path / "candidate"
    for sequence in ("AA_00", "BB_00"):
        _write(candidate / f"{sequence}.txt", _TRUTH_ROWS)

    result = run_guarded_tournament(
        raw,
        truth,
        tmp_path / "out",
        candidates=(("perfect", candidate),),
        fold_count=2,
        expected_sequence_count=2,
        bootstrap_samples=200,
        copy_selected=False,
    )

    assert result.selected_candidate == "perfect"
    selected = next(row for row in result.rows if row.selected)
    assert selected.eligible
    assert selected.mean_cv_hota_gain_vs_raw > 0.0
    assert selected.paired_hota_gain_ci_low > 0.0
    summary = json.loads(
        (tmp_path / "out" / "tournament_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["selection_status"] == "transformed_candidate_selected"
    assert (tmp_path / "out" / "tournament_ranking.csv").is_file()
    assert (tmp_path / "out" / "sequence_deltas.csv").is_file()
    assert (tmp_path / "out" / "provenance.json").is_file()


def test_guarded_tournament_falls_back_to_raw(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    raw = tmp_path / "raw"
    candidate = tmp_path / "candidate"
    for sequence in ("AA_00", "BB_00"):
        _write(truth / f"{sequence}.txt", _TRUTH_ROWS)
        _write(raw / f"{sequence}.txt", _TRUTH_ROWS)
        _write(candidate / f"{sequence}.txt", _SWITCH_ROWS)

    result = run_guarded_tournament(
        raw,
        truth,
        tmp_path / "out",
        candidates=(("regression", candidate),),
        fold_count=2,
        expected_sequence_count=2,
        bootstrap_samples=100,
        copy_selected=False,
    )

    assert result.selected_candidate == "raw"
    regressed = next(row for row in result.rows if row.name == "regression")
    assert not regressed.eligible
    assert any(
        "CODABENCH_HOTA gain" in reason
        for reason in regressed.rejection_reasons
    )
    summary = json.loads(
        (tmp_path / "out" / "tournament_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["selection_status"] == "raw_fallback"


def test_guarded_tournament_rejects_incomplete_candidate(tmp_path: Path) -> None:
    truth, raw = _two_sequence_fixture(tmp_path)
    candidate = tmp_path / "candidate"
    _write(candidate / "AA_00.txt", _TRUTH_ROWS)

    with pytest.raises(ValueError, match="missing 1 selected sequence files"):
        run_guarded_tournament(
            raw,
            truth,
            tmp_path / "out",
            candidates=(("incomplete", candidate),),
            fold_count=2,
            expected_sequence_count=2,
            bootstrap_samples=0,
            copy_selected=False,
        )
