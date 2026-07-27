from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_bottleneck import (
    BOTTLENECK_UNKNOWN,
    BottleneckConfig,
    build_bottleneck_summary,
    classify_gap_row,
    write_bottleneck_outputs,
)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("mixture_mse_3d_m2", np.float32(np.inf)),
        ("mixture_mse_3d_m2", np.float64(-np.inf)),
        ("reservoir_oracle_all_mse_3d_m2", np.float32(np.inf)),
    ],
)
def test_classify_gap_row_treats_nonfinite_metrics_as_missing(
    column: str,
    value: object,
) -> None:
    row = {
        "mixture_mse_3d_m2": 5.0,
        "reservoir_oracle_all_mse_3d_m2": 2.0,
        column: value,
    }

    result = classify_gap_row(row)

    assert result[column] is None
    assert result["primary_bottleneck"] == BOTTLENECK_UNKNOWN


def test_write_bottleneck_outputs_emits_strict_json_for_numpy_values(
    tmp_path: Path,
) -> None:
    annotated = pd.DataFrame(
        {
            "primary_bottleneck": ["assignment_limited"],
            "recommended_action": ["improve_assignment"],
            "assignment_gap_mse_3d_m2": [2.0],
            "topk_recall_gap_mse_3d_m2": [1.0],
            "reservoir_oracle_all_mse_3d_m2": [3.0],
            "upstream_nonfinite": pd.Series([np.float32(np.inf)], dtype=object),
            "upstream_vector": pd.Series(
                [np.asarray([1.0, np.nan], dtype=np.float32)],
                dtype=object,
            ),
        }
    )
    summary_json = tmp_path / "summary.json"

    write_bottleneck_outputs(
        annotated,
        output_csv=tmp_path / "annotated.csv",
        summary_json=summary_json,
    )

    text = summary_json.read_text(encoding="utf-8")
    payload = json.loads(
        text,
        parse_constant=lambda token: pytest.fail(f"non-standard JSON token: {token}"),
    )
    worst = payload["worst_assignment_gap"]

    assert "NaN" not in text
    assert "Infinity" not in text
    assert worst["upstream_nonfinite"] is None
    assert worst["upstream_vector"] == [1.0, None]


def test_bottleneck_summary_ignores_nonfinite_worst_metrics() -> None:
    annotated = pd.DataFrame(
        {
            "sequence_id": ["positive_inf", "negative_inf", "finite"],
            "primary_bottleneck": ["unknown", "unknown", "assignment_limited"],
            "recommended_action": [
                "inspect_missing_metrics",
                "inspect_missing_metrics",
                "improve_assignment",
            ],
            "assignment_gap_mse_3d_m2": [np.inf, -np.inf, 12.0],
            "topk_recall_gap_mse_3d_m2": [np.nan, np.inf, -np.inf],
            "reservoir_oracle_all_mse_3d_m2": [np.inf, 3.0, 4.0],
        }
    )

    summary = build_bottleneck_summary(annotated, config=BottleneckConfig())

    assert summary["worst_assignment_gap"]["sequence_id"] == "finite"
    assert summary["worst_assignment_gap"]["assignment_gap_mse_3d_m2"] == 12.0
    assert summary["worst_topk_recall_gap"] == {}
    assert summary["worst_reservoir_ceiling"]["sequence_id"] == "finite"
