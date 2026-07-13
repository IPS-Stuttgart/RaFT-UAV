from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile
import zipfile

import pytest

from raft_uav.mmuad.layout import inspect_mmuad_layout


@pytest.mark.parametrize("archive_kind", ["zip", "tar"])
def test_layout_inspector_keeps_malformed_archived_topic_map_inventoryable(
    tmp_path: Path,
    archive_kind: str,
) -> None:
    malformed_topic_map = b'{"exports":[{"kind":"truth","path":"truth.csv"}],"note":"\xff"}'
    if archive_kind == "zip":
        archive_path = tmp_path / "dataset.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("val/seq_bad/topic_map.json", malformed_topic_map)
            archive.writestr("val/seq_bad/candidates.csv", "time_s,x_m,y_m,z_m\n0,1,2,3\n")
    else:
        archive_path = tmp_path / "dataset.tar"
        with tarfile.open(archive_path, "w") as archive:
            for name, payload in (
                ("val/seq_bad/topic_map.json", malformed_topic_map),
                ("val/seq_bad/candidates.csv", b"time_s,x_m,y_m,z_m\n0,1,2,3\n"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))

    summary = inspect_mmuad_layout(archive_path)

    assert summary["archive_count"] == 1
    assert summary["archive_member_count"] == 2
    assert summary["category_counts"]["json_metadata"] == 1
    assert summary["category_counts"]["candidate_or_point_table"] == 1
    candidate = summary["sequence_candidates"][0]
    assert candidate["sequence_id"] == "seq_bad"
    assert candidate["has_truth_or_labels"] is False
