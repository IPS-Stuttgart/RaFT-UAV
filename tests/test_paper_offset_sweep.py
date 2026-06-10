from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from raft_uav.diagnostics import paper_offset_sweep


def test_offset_sweep_best_json_converts_dataframe_nan_to_null(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_evaluate_offset_pair(**kwargs):
        rf_residual_s = float(kwargs["rf_residual_s"])
        row = {
            "flight": "Opt1",
            "rf_residual_offset_s": rf_residual_s,
            "radar_residual_offset_s": float(kwargs["radar_residual_s"]),
            "rf_clock_offset_s": float(kwargs["rf_clock_offset_s"]),
            "radar_clock_offset_s": float(kwargs["radar_clock_offset_s"]),
            "paper_parity_score": rf_residual_s,
            "count_abs_delta_total": 0,
            "kf_all_steps_mean_delta_m": 0.0,
            "kf_all_steps_mean_abs_delta_m": 0.0,
        }
        if rf_residual_s > 0.0:
            row["worse_candidate_only_metric_m"] = 12.0
        return row

    monkeypatch.setattr(paper_offset_sweep, "_evaluate_offset_pair", fake_evaluate_offset_pair)

    result = paper_offset_sweep.run_offset_sweep(
        dataset_root=tmp_path,
        flight="Opt1",
        output_dir=tmp_path / "out",
        variant="rerun",
        rf_clock_offset_s=0.0,
        radar_clock_offset_s=0.0,
        rf_residual_grid_s=np.array([0.0, 1.0]),
        radar_residual_grid_s=np.array([0.0]),
        range_gate_m=850.0,
        nis_gate_probability=0.997,
        truth_time_gate_s=2.0,
        acceleration_std_mps2=4.0,
        enu_origin="lw1",
        enu_origin_lla=None,
        lw1_origin_lla=None,
        origin_config=None,
        empirical_covariance=True,
    )

    payload_text = Path(result["best_json"]).read_text(encoding="utf-8")
    payload = json.loads(payload_text)

    assert "NaN" not in payload_text
    assert payload["best"]["worse_candidate_only_metric_m"] is None
