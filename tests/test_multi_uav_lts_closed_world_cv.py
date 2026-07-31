from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts import closed_world_cv


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _candidate(
    config_id: str,
    *,
    is_raw: bool,
    hota: float,
    mota: float,
    idf1: float,
    grid_index: int,
) -> dict[str, object]:
    return {
        "config_id": config_id,
        "is_raw": is_raw,
        "mean_codabench_hota": hota,
        "std_codabench_hota": 0.0,
        "mean_codabench_mota": mota,
        "mean_codabench_idf1": idf1,
        "grid_index": grid_index,
    }


def _rank(
    *,
    raw: tuple[float, float, float],
    candidate: tuple[float, float, float],
) -> list[dict[str, object]]:
    return closed_world_cv._rank_candidates(
        [
            _candidate(
                "raw",
                is_raw=True,
                hota=raw[0],
                mota=raw[1],
                idf1=raw[2],
                grid_index=-1,
            ),
            _candidate(
                "closed_world",
                is_raw=False,
                hota=candidate[0],
                mota=candidate[1],
                idf1=candidate[2],
                grid_index=0,
            ),
        ],
        max_mota_drop=0.005,
        max_idf1_drop=0.005,
        min_hota_gain=0.0,
    )


def test_raw_baseline_wins_when_postprocessing_regresses() -> None:
    rows = _rank(raw=(0.8, 0.8, 0.8), candidate=(0.79, 0.82, 0.82))

    assert rows[0]["is_raw"]
    assert rows[0]["eligible"]
    assert rows[1]["hota_gain_vs_raw"] == pytest.approx(-0.01)


def test_metric_floor_rejects_hota_gain_with_large_mota_drop() -> None:
    rows = _rank(raw=(0.8, 0.8, 0.8), candidate=(0.9, 0.7, 0.9))

    candidate = next(row for row in rows if not row["is_raw"])
    assert not candidate["eligible"]
    assert rows[0]["is_raw"]


def test_guarded_hota_gain_selects_closed_world_candidate() -> None:
    rows = _rank(raw=(0.8, 0.8, 0.8), candidate=(0.82, 0.801, 0.81))

    assert not rows[0]["is_raw"]
    assert rows[0]["eligible"]
    assert rows[0]["hota_gain_vs_raw"] == pytest.approx(0.02)


def test_ranking_requires_exactly_one_raw_candidate() -> None:
    candidate = _candidate(
        "closed_world",
        is_raw=False,
        hota=0.8,
        mota=0.8,
        idf1=0.8,
        grid_index=0,
    )

    with pytest.raises(ValueError, match="exactly one raw candidate"):
        closed_world_cv._rank_candidates(
            [candidate],
            max_mota_drop=0.005,
            max_idf1_drop=0.005,
            min_hota_gain=0.0,
        )


def test_closed_world_cv_improves_crossing_id_switch(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    truth = tmp_path / "truth"
    labels = tmp_path / "labels"
    truth_text = (
        "1,7,0,0,10,10,1,1,1\n"
        "1,9,100,0,10,10,1,1,1\n"
        "2,7,10,0,10,10,1,1,1\n"
        "2,9,90,0,10,10,1,1,1\n"
        "3,7,20,0,10,10,1,1,1\n"
        "3,9,80,0,10,10,1,1,1\n"
    )
    prediction_text = (
        "1,1,0,0,10,10,1,1,1\n"
        "1,2,100,0,10,10,1,1,1\n"
        "2,1,10,0,10,10,1,1,1\n"
        "2,2,90,0,10,10,1,1,1\n"
        "3,1,80,0,10,10,1,1,1\n"
        "3,2,20,0,10,10,1,1,1\n"
    )
    first_frame = (
        "1,7,0,0,10,10,1,1,1\n"
        "1,9,100,0,10,10,1,1,1\n"
    )
    for sequence in ("C_00", "T_00"):
        _write(predictions / f"{sequence}.txt", prediction_text)
        _write(truth / f"{sequence}.txt", truth_text)
        _write(labels / f"{sequence}.txt", first_frame)

    rows = closed_world_cv.run_closed_world_cv(
        predictions,
        truth,
        labels,
        tmp_path / "cv",
        fold_count=2,
        max_gaps=(2,),
        max_costs=(2.0,),
        source_continuity_bonuses=(0.2,),
        coast_options=(False,),
    )

    assert not rows[0].is_raw
    assert rows[0].mean_codabench_hota == pytest.approx(1.0)
    raw = next(row for row in rows if row.is_raw)
    assert rows[0].mean_codabench_hota > raw.mean_codabench_hota


def test_materialize_predictions_allows_source_destination_alias(tmp_path: Path) -> None:
    best = tmp_path / "best_predictions"
    _write(best / "C_00.txt", "1,7,0,0,10,10,1,1,1\n")

    closed_world_cv._materialize_predictions(best, best, ("C_00",))

    assert (best / "C_00.txt").read_text(encoding="utf-8") == (
        "1,7,0,0,10,10,1,1,1\n"
    )
