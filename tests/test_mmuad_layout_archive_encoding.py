from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile
import zipfile

from raft_uav.mmuad.layout import inspect_mmuad_layout

_INVALID_TOPIC_MAP = b"\xff\xfe\x00not-utf8"
_CANDIDATES = b"time_s,x_m,y_m,z_m\n0,1,2,3\n"


def _assert_malformed_topic_map_is_inventoried(summary: dict[str, object]) -> None:
    assert summary["archive_count"] == 1
    assert summary["archive_member_count"] == 2
    assert summary["category_counts"] == {
        "candidate_or_point_table": 1,
        "json_metadata": 1,
    }
    sequence = summary["sequence_candidates"][0]
    assert sequence["sequence_id"] == "seq001"
    assert sequence["has_topic_map_export"] is False
    assert sequence["has_truth_or_labels"] is False


def test_layout_inspector_tolerates_non_utf8_zip_topic_map(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("seq001/topic_map.json", _INVALID_TOPIC_MAP)
        archive.writestr("seq001/candidates.csv", _CANDIDATES)

    summary = inspect_mmuad_layout(archive_path)

    _assert_malformed_topic_map_is_inventoried(summary)


def _add_tar_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, BytesIO(payload))


def test_layout_inspector_tolerates_non_utf8_tar_topic_map(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        _add_tar_member(archive, "seq001/topic_map.yaml", _INVALID_TOPIC_MAP)
        _add_tar_member(archive, "seq001/candidates.csv", _CANDIDATES)

    summary = inspect_mmuad_layout(archive_path)

    _assert_malformed_topic_map_is_inventoried(summary)
