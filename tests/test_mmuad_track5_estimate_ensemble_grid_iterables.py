from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import raft_uav.mmuad.track5_estimate_ensemble_grid as grid_module
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble_grid import (
    evaluate_estimate_ensemble_weight_grid,
)
from raft_uav.mmuad.track5_estimate_ensemble_grid import (
    write_estimate_ensemble_weight_grid_outputs,
)


def _write_estimate(path: Path, value: float) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [value],
            "state_y_m": [value],
            "state_z_m": [value],
        }
    ).to_csv(path, index=False)


def _case(tmp_path: Path) -> tuple[list[EstimateInput], pd.DataFrame, pd.DataFrame]:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    third = tmp_path / "third.csv"
    _write_estimate(first, 0.0)
    _write_estimate(second, 10.0)
    _write_estimate(third, 100.0)

    inputs = [
        parse_estimate_spec(f"first={first}"),
        parse_estimate_spec(f"second={second}"),
        parse_estimate_spec(f"third={third}"),
    ]
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "x_m": [10.0],
            "y_m": [10.0],
            "z_m": [10.0],
            "class_name": ["0"],
        }
    )
    return inputs, template, truth


def test_weight_grid_generator_is_reused_for_every_aggregation_policy(
    tmp_path: Path,
) -> None:
    inputs, template, truth = _case(tmp_path)
    weight_grid = (weights for weights in ((1.0 / 3.0,) * 3,))

    summary, _, best_weights = evaluate_estimate_ensemble_weight_grid(
        inputs,
        template=template,
        truth=truth,
        weight_grid=weight_grid,
        default_classification=0,
        aggregation_policies=("weighted-mean", "weighted-median"),
    )

    assert len(summary) == 2
    assert set(summary["aggregation_policy"]) == {"weighted-mean", "weighted-median"}
    assert summary.iloc[0]["aggregation_policy"] == "weighted-median"
    assert summary.iloc[0]["pose_mse"] == pytest.approx(0.0)
    assert best_weights == pytest.approx((1.0 / 3.0,) * 3)


def test_policy_generator_is_reused_for_evaluation_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, template, truth = _case(tmp_path)
    seen_policies: list[tuple[str, ...]] = []

    def fake_evaluate(
        estimate_inputs: Any,
        *,
        aggregation_policies: Any,
        **_: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, ...]]:
        assert tuple(estimate_inputs) == tuple(inputs)
        policies = tuple(aggregation_policies)
        seen_policies.append(policies)
        summary = pd.DataFrame(
            [
                {
                    "aggregation_policy": policies[-1],
                    "trim_fraction": 0.2,
                    "pose_mse": 0.0,
                    "p95_error_m": 0.0,
                    "max_error_m": 0.0,
                }
            ]
        )
        return summary, pd.DataFrame(), (1.0 / 3.0,) * 3

    def fake_write_outputs(*, output_dir: Path, **_: Any) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"official_zip": output_dir / "ug2_submission.zip"}

    monkeypatch.setattr(grid_module, "evaluate_estimate_ensemble_weight_grid", fake_evaluate)
    monkeypatch.setattr(grid_module, "write_track5_estimate_ensemble_outputs", fake_write_outputs)

    paths = write_estimate_ensemble_weight_grid_outputs(
        estimate_inputs=inputs,
        template=template,
        truth=truth,
        weight_grid=(weights for weights in ((1.0 / 3.0,) * 3,)),
        output_dir=tmp_path / "out",
        default_classification=0,
        aggregation_policies=(
            policy for policy in ("weighted-mean", "weighted-median")
        ),
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert seen_policies == [("weighted-mean", "weighted-median")]
    assert manifest["aggregation_policies"] == ["weighted-mean", "weighted-median"]
    assert manifest["grid_row_count"] == 1
    assert manifest["best_aggregation_policy"] == "weighted-median"
