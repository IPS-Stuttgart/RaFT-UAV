from __future__ import annotations

import numpy as np

from raft_uav.diagnostics.nis_reliability import (
    nis_reliability_summary,
    read_nis_diagnostics,
)


def test_nis_diagnostics_preserve_opaque_sequence_ids(tmp_path) -> None:
    diagnostics = tmp_path / "diagnostics.csv"
    diagnostics.write_text(
        "sequence_id,source,measurement_dim,nis\n"
        "001,rf,2,1.0\n"
        "1,rf,2,3.0\n"
        "NA,rf,2,5.0\n",
        encoding="utf-8",
    )

    frame = read_nis_diagnostics([diagnostics])

    assert frame["sequence_id"].tolist() == ["001", "1", "NA"]
    summary = nis_reliability_summary(
        frame,
        group_columns=("sequence_id", "source", "measurement_dim"),
        gate_probabilities=(0.95,),
    ).set_index("sequence_id")

    assert set(summary.index) == {"001", "1", "NA"}
    assert summary["count"].astype(int).to_dict() == {"001": 1, "1": 1, "NA": 1}
    assert np.isclose(float(summary.loc["001", "nis_mean"]), 1.0)
    assert np.isclose(float(summary.loc["1", "nis_mean"]), 3.0)
    assert np.isclose(float(summary.loc["NA", "nis_mean"]), 5.0)
