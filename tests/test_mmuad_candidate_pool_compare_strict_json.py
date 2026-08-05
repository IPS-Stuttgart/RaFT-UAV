from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_pool_compare import (
    write_candidate_pool_compare_outputs,
)


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def test_candidate_pool_compare_summary_uses_strict_json(tmp_path: Path) -> None:
    pooled = pd.DataFrame(
        [
            {
                "pool_label": "candidate",
                "nan_metric": np.nan,
                "positive_inf_metric": np.inf,
                "negative_inf_metric": -np.inf,
                "missing_metric": pd.NA,
                "numpy_count": np.int64(3),
            }
        ]
    )

    paths = write_candidate_pool_compare_outputs(
        output_dir=tmp_path,
        frame_rows=pd.DataFrame(),
        pooled_summary=pooled,
        by_sequence=pd.DataFrame(),
        by_reference_branch=pd.DataFrame(),
    )

    summary_path = Path(paths["summary_json"])
    payload = json.loads(
        summary_path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
    )

    row = payload["pooled"][0]
    assert row["pool_label"] == "candidate"
    assert row["nan_metric"] is None
    assert row["positive_inf_metric"] is None
    assert row["negative_inf_metric"] is None
    assert row["missing_metric"] is None
    assert row["numpy_count"] == 3
