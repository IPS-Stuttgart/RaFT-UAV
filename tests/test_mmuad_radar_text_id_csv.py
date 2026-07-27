from __future__ import annotations

import pandas as pd

from raft_uav.mmuad import radar as radar_module


def test_radar_csv_preserves_na_like_identifier_tokens(tmp_path) -> None:
    path = tmp_path / "radar.csv"
    path.write_text(
        "sequence_id,track_id,class_name,range_m,azimuth_deg,time_s\n"
        "NA,N/A,null,100,0,1\n"
        ",,,200,5,2\n"
        "seq,track,uav,NA,10,3\n",
        encoding="utf-8",
    )

    rows = radar_module._read_csv_preserving_text_ids(path)

    assert rows.loc[0, "sequence_id"] == "NA"
    assert rows.loc[0, "track_id"] == "N/A"
    assert rows.loc[0, "class_name"] == "null"
    assert pd.isna(rows.loc[1, "sequence_id"])
    assert pd.isna(rows.loc[1, "track_id"])
    assert pd.isna(rows.loc[1, "class_name"])
    assert pd.isna(rows.loc[2, "range_m"])
    assert all(
        str(rows[column].dtype).startswith("string")
        for column in ("sequence_id", "track_id", "class_name")
    )
