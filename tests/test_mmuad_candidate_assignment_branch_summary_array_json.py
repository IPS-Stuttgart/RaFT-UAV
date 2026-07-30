from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_assignment_branch_summary import (
    write_candidate_assignment_branch_summary,
)


def test_branch_summary_writes_array_and_nullable_provenance_as_strict_json(
    tmp_path: Path,
) -> None:
    paths = write_candidate_assignment_branch_summary(
        output_dir=tmp_path,
        summary=pd.DataFrame(),
        provenance={
            "array": np.array([1.0, np.nan, np.inf]),
            "zero_dimensional": np.array(np.nan),
            "nullable": pd.NA,
            "scalar": np.int64(3),
        },
    )

    payload = json.loads(
        Path(paths["branch_summary_json"]).read_text(encoding="utf-8")
    )
    assert payload["array"] == [1.0, None, None]
    assert payload["zero_dimensional"] is None
    assert payload["nullable"] is None
    assert payload["scalar"] == 3
    json.dumps(payload, allow_nan=False)
