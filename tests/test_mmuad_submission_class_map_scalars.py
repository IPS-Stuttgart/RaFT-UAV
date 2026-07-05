from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad import submission


def test_load_sequence_class_map_accepts_numpy_scalar_uav_type(monkeypatch, tmp_path: Path) -> None:
    class_map_path = tmp_path / "class_map.csv"
    class_map_path.write_text("sequence_id,uav_type\nseq1,2\n", encoding="utf-8")

    def fake_read_csv(path: Path | str) -> pd.DataFrame:
        assert Path(path) == class_map_path
        return pd.DataFrame(
            {
                "sequence_id": pd.Series(["seq1"], dtype=object),
                "uav_type": pd.Series([np.int64(2)], dtype=object),
            }
        )

    monkeypatch.setattr(submission._impl.pd, "read_csv", fake_read_csv)

    assert submission.load_sequence_class_map(class_map_path) == {"seq1": "2"}
