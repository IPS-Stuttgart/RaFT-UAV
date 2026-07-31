from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from raft_uav.multi_uav_lts import closed_world_cv


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _metrics(hota: float, mota: float, idf1: float) -> SimpleNamespace:
    return SimpleNamespace(
        codabench_hota=hota,
        codabench_mota=mota,
        codabench_idf1=idf1,
        hota=hota,
        deta=hota,
        assa=hota,
        loca=1.0,
        mota=mota,
        idf1=idf1,
        predicted_detections=4,
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    raw_path: Path,
    *,
    raw_metrics: SimpleNamespace,
    candidate_metrics: SimpleNamespace,
) -> None:
    def fake_evaluate(
        prediction_path: Path,
        truth_dir: Path,
        *,
        sequences=(),
    ) -> SimpleNamespace:
        del truth_dir, sequences
        return raw_metrics if prediction_path == raw_path else candidate_metrics

    def fake_postprocess(
        prediction_path: Path,
        first_frame_label_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> SimpleNamespace:
        del first_frame_label_dir, kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        for source in prediction_path.glob("*.txt"):
            (output_dir / source.name).write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )
        return SimpleNamespace(
            output_rows=4,
            dropped_candidate_rows=0,
            absorbed_source_switches=2,
            coasted_rows=0,
        )

    # Several repository regressions reload compatibility packages. Patch the
    # exact global namespace used by the function under test rather than a
    # potentially replaced package attribute.
    function_globals = closed_world_cv.run_closed_world_cv.__globals__
    monkeypatch.setitem(
        function_globals, "evaluate_lts_predictions", fake_evaluate
    )
    monkeypatch.setitem(
        function_globals, "postprocess_closed_world", fake_postprocess
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    predictions = tmp_path / "predictions"
    truth = tmp_path / "truth"
    labels = tmp_path / "labels"
    for sequence in ("C_00", "T_00"):
        text = "1,7,0,0,10,10,1,1,1\n2,7,1,0,10,10,1,1,1\n"
        _write(predictions / f"{sequence}.txt", text)
        _write(truth / f"{sequence}.txt", text)
        _write(labels / f"{sequence}.txt", text.splitlines()[0] + "\n")
    return predictions, truth, labels


def _run(tmp_path: Path) -> tuple[closed_world_cv.ClosedWorldCVRow, ...]:
    predictions, truth, labels = _inputs(tmp_path)
    return closed_world_cv.run_closed_world_cv(
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


def test_raw_baseline_wins_when_postprocessing_regresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions, _, _ = _inputs(tmp_path)
    _install_fakes(
        monkeypatch,
        predictions,
        raw_metrics=_metrics(0.8, 0.8, 0.8),
        candidate_metrics=_metrics(0.79, 0.82, 0.82),
    )

    rows = _run(tmp_path)

    assert rows[0].is_raw
    assert rows[0].selected
    assert (tmp_path / "cv/best_predictions/C_00.txt").exists()


def test_metric_floor_rejects_hota_gain_with_large_mota_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions, _, _ = _inputs(tmp_path)
    _install_fakes(
        monkeypatch,
        predictions,
        raw_metrics=_metrics(0.8, 0.8, 0.8),
        candidate_metrics=_metrics(0.9, 0.7, 0.9),
    )

    rows = _run(tmp_path)

    candidate = next(row for row in rows if not row.is_raw)
    assert not candidate.eligible
    assert rows[0].is_raw


def test_guarded_hota_gain_selects_closed_world_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions, _, _ = _inputs(tmp_path)
    _install_fakes(
        monkeypatch,
        predictions,
        raw_metrics=_metrics(0.8, 0.8, 0.8),
        candidate_metrics=_metrics(0.82, 0.801, 0.81),
    )

    rows = _run(tmp_path)

    assert not rows[0].is_raw
    assert rows[0].eligible
    assert rows[0].hota_gain_vs_raw == pytest.approx(0.02)


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
