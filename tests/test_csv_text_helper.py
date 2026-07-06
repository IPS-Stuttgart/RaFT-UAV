from pathlib import Path

from raft_uav.mmuad.csv_text import SEQUENCE_ID_COLUMNS


def test_sequence_id_columns_cover_official_name() -> None:
    assert "Sequence" in SEQUENCE_ID_COLUMNS
    assert "sequence_id" in SEQUENCE_ID_COLUMNS
