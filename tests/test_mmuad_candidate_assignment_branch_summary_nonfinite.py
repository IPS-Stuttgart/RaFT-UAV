from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_assignment_branch_summary import (
    build_candidate_assignment_branch_summary,
    write_candidate_assignment_branch_summary,
)


def test_branch_summary_ignores_nonfinite_numeric_measurements() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "state_error_3d_m": [3.0, np.inf, -np.inf],
            "oracle_error_3d_m": [1.0, np.inf, -np.inf],
            "dominant_error_3d_m": [2.0, np.inf, -np.inf],
            "state_regret_m": [2.0, np.inf, -np.inf],
            "dominant_regret_m": [1.0, np.inf, -np.inf],
            "oracle_mixture_weight": [0.25, np.inf, -np.inf],
            "oracle_weight_rank": [2.0, np.inf, -np.inf],
            "candidate_count": [4.0, np.inf, -np.inf],
            "dominant_is_oracle": [True, False, False],
            "oracle_in_topk_by_weight": [True, False, False],
        }
    )

    summary = build_candidate_assignment_branch_summary(rows)
    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["group_label"] == "__all__")
    ].iloc[0]

    assert pooled["state_error_3d_m_mse"] == 9.0
    assert pooled["oracle_error_3d_m_mse"] == 1.0
    assert pooled["dominant_error_3d_m_mse"] == 4.0
    assert pooled["state_regret_m_mean"] == 2.0
    assert pooled["dominant_regret_m_mean"] == 1.0
    assert pooled["oracle_mixture_weight_mean"] == 0.25
    assert pooled["oracle_weight_rank_p50"] == 2.0
    assert pooled["candidate_count_mean"] == 4.0


def test_branch_summary_writes_strict_json_for_nonfinite_values(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "state_error_3d_m_mse": np.nan,
                "oracle_error_3d_m_mse": np.inf,
            }
        ]
    )

    paths = write_candidate_assignment_branch_summary(
        output_dir=tmp_path,
        summary=summary,
        provenance={"nested": [np.float64(-np.inf)]},
    )

    text = Path(paths["branch_summary_json"]).read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["summary"][0]["state_error_3d_m_mse"] is None
    assert payload["summary"][0]["oracle_error_3d_m_mse"] is None
    assert payload["nested"] == [None]
    json.dumps(payload, allow_nan=False)
