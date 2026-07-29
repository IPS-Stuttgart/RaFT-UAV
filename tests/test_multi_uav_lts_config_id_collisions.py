from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.fixed_population_cv import run_fixed_population_cv
from raft_uav.multi_uav_lts.fixed_population_grid import run_fixed_population_grid

_THRESHOLDS = (0.5001, 0.5004)
_SHIFT_FOR_IOU_0_50025 = 3.331111481419764
_BASE_CONFIG_ID = "seed0p500_gap0_cost1p000_interp0"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_collision_case(
    truth_dir: Path,
    label_dir: Path,
    prediction_dir: Path,
    *,
    sequence: str,
    offset: int,
) -> str:
    expected = (
        f"1,7,{offset},0,10,10,1,1,1\n"
        f"2,7,{offset + 1},0,10,10,1,1,1\n"
    )
    _write(truth_dir / f"{sequence}.txt", expected)
    _write(
        label_dir / f"{sequence}.txt",
        f"1,7,{offset},0,10,10,1,1,1\n",
    )
    _write(
        prediction_dir / f"{sequence}.txt",
        f"1,1,{offset + _SHIFT_FOR_IOU_0_50025},0,10,10,1,1,1\n"
        f"2,1,{offset + 1},0,10,10,1,1,1\n",
    )
    return expected


def test_grid_keeps_colliding_rounded_config_outputs_separate(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "grid"
    expected = _write_collision_case(
        truth,
        labels,
        predictions,
        sequence="C_00",
        offset=0,
    )

    rows = run_fixed_population_grid(
        predictions,
        truth,
        labels,
        output,
        min_seed_ious=_THRESHOLDS,
        relink_max_gaps=(0,),
        relink_max_costs=(1.0,),
        interpolation_options=(False,),
    )

    assert rows[0].min_seed_iou == _THRESHOLDS[0]
    assert rows[0].codabench_hota == 1.0
    assert tuple(row.config_id for row in rows) == (
        _BASE_CONFIG_ID,
        f"{_BASE_CONFIG_ID}__2",
    )
    assert Path(rows[0].prediction_dir).read_text(encoding="utf-8") == expected
    assert (output / "best_predictions" / "C_00.txt").read_text(
        encoding="utf-8"
    ) == expected


def test_cv_keeps_colliding_rounded_config_outputs_separate(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "cv"
    expected_by_sequence = {
        sequence: _write_collision_case(
            truth,
            labels,
            predictions,
            sequence=sequence,
            offset=offset,
        )
        for sequence, offset in (("C_00", 0), ("C_01", 20))
    }

    rows = run_fixed_population_cv(
        predictions,
        truth,
        labels,
        output,
        fold_count=2,
        seed=0,
        min_seed_ious=_THRESHOLDS,
        relink_max_gaps=(0,),
        relink_max_costs=(1.0,),
        interpolation_options=(False,),
    )

    assert rows[0].min_seed_iou == _THRESHOLDS[0]
    assert rows[0].mean_codabench_hota == 1.0
    assert tuple(row.config_id for row in rows) == (
        _BASE_CONFIG_ID,
        f"{_BASE_CONFIG_ID}__2",
    )
    for sequence, expected in expected_by_sequence.items():
        assert Path(rows[0].prediction_dir, f"{sequence}.txt").read_text(
            encoding="utf-8"
        ) == expected
        assert (output / "best_predictions" / f"{sequence}.txt").read_text(
            encoding="utf-8"
        ) == expected
