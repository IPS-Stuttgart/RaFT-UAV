from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_class_conditioned_ensemble import (
    build_class_conditioned_estimate_ensemble,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def test_class_conditioned_ensemble_preserves_official_sequence_separators(
    tmp_path: Path,
) -> None:
    """Sequence ids are match keys, so spaces/slashes must not be filename-sanitized."""

    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq/A", "seq/A", "seq B", "seq B"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [0.0, 1.0, 20.0, 21.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(a_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq/A", "seq/A", "seq B", "seq B"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [8.0, 9.0, 10.0, 11.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(b_csv, index=False)
    template = pd.DataFrame(
        {
            "Sequence": ["seq/A", "seq/A", "seq B", "seq B"],
            "Timestamp": [0.0, 1.0, 0.0, 1.0],
        }
    )
    config = {
        "global_weights": {"a": 0.5, "b": 0.5},
        "class_weights": {
            "0": {"a": 1.0, "b": 0.0},
            "1": {"a": 0.0, "b": 1.0},
        },
    }

    estimates, diagnostics = build_class_conditioned_estimate_ensemble(
        [EstimateInput("a", a_csv), EstimateInput("b", b_csv)],
        template=template,
        class_map={"seq/A": "0", "seq B": "1"},
        weight_config=config,
    )

    by_key = estimates.set_index(["sequence_id", "time_s"])["state_x_m"].to_dict()
    assert by_key[("seq/A", 0.0)] == 0.0
    assert by_key[("seq/A", 1.0)] == 1.0
    assert by_key[("seq B", 0.0)] == 10.0
    assert by_key[("seq B", 1.0)] == 11.0
    assert set(estimates["class_conditioned_weight_source"]) == {"class"}
    assert set(diagnostics["class_conditioned_ensemble_class"]) == {"0", "1"}
