from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import apply_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import load_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import main as ensemble_main


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2],
        }
    )


def _estimate(rows_x: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "state_x_m": list(rows_x),
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )


def test_load_estimate_weight_config_accepts_weight_search_schema(tmp_path: Path) -> None:
    config = tmp_path / "best_weights.json"
    config.write_text(json.dumps({"weights": {"good": 0.75, "bad": 0.25}}), encoding="utf-8")

    assert load_estimate_weight_config(config) == {"good": 0.75, "bad": 0.25}


def test_load_estimate_weight_config_accepts_direct_mapping(tmp_path: Path) -> None:
    config = tmp_path / "weights.json"
    config.write_text(json.dumps({"good": 1.0, "bad": 0.0}), encoding="utf-8")

    assert load_estimate_weight_config(config) == {"good": 1.0, "bad": 0.0}


def test_apply_estimate_weight_config_overrides_inline_weights(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    inputs = [EstimateInput("good", good, 0.0), EstimateInput("bad", bad, 1.0)]

    updated = apply_estimate_weight_config(inputs, {"good": 1.0, "bad": 0.0})

    assert [item.weight for item in updated] == [1.0, 0.0]


def test_apply_estimate_weight_config_can_keep_missing_inline_weight(tmp_path: Path) -> None:
    inputs = [EstimateInput("good", tmp_path / "good.csv", 0.25)]

    updated = apply_estimate_weight_config(inputs, {}, missing_policy="keep")

    assert updated == inputs


def test_apply_estimate_weight_config_rejects_missing_label_by_default(tmp_path: Path) -> None:
    inputs = [EstimateInput("good", tmp_path / "good.csv", 1.0)]

    with pytest.raises(ValueError, match="missing ensemble weights"):
        apply_estimate_weight_config(inputs, {"other": 1.0})


def test_track5_estimate_ensemble_cli_uses_weights_json(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    weights_json = tmp_path / "weights.json"
    output_dir = tmp_path / "out"
    _estimate((1.0, 2.0)).to_csv(good_csv, index=False)
    _estimate((101.0, 102.0)).to_csv(bad_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )
    weights_json.write_text(json.dumps({"weights": {"good": 1.0, "bad": 0.0}}), encoding="utf-8")

    status = ensemble_main(
        [
            "--estimate-csv",
            f"good={good_csv}@0",
            "--estimate-csv",
            f"bad={bad_csv}@1",
            "--weights-json",
            str(weights_json),
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output_dir / "mmuad_track5_ensemble_estimates.csv")
    manifest = json.loads((output_dir / "mmuad_track5_ensemble_manifest.json").read_text())
    assert estimates["state_x_m"].tolist() == [1.0, 2.0]
    assert [item["weight"] for item in manifest["estimate_inputs"]] == [1.0, 0.0]
