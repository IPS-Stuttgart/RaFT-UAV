from __future__ import annotations

from pathlib import Path

from raft_uav.mmuad.rosbag_bridge import load_topic_map_payload
from raft_uav.mmuad.splits import load_split_manifest
from raft_uav.mmuad.submission import load_sequence_class_map


def test_mmuad_yaml_exports_are_supported_by_runtime_dependencies(tmp_path: Path) -> None:
    """YAML support must work in a normal package install, not only in dev envs."""

    class_map = tmp_path / "classes.yaml"
    class_map.write_text(
        "class_map:\n"
        "  seq_yaml:\n"
        "    uav_type: Mavic3\n",
        encoding="utf-8",
    )
    split_manifest = tmp_path / "splits.yaml"
    split_manifest.write_text(
        "splits:\n"
        "  val:\n"
        "    sequences:\n"
        "      - sequence_id: seq_yaml\n",
        encoding="utf-8",
    )
    topic_map = tmp_path / "topic_map.yaml"
    topic_map.write_text(
        "schema: raft-uav-mmuad-topic-map-v1\n"
        "sequence_id: seq_yaml\n"
        "exports: []\n",
        encoding="utf-8",
    )

    assert load_sequence_class_map(class_map) == {"seq_yaml": "Mavic3"}
    assert load_split_manifest(split_manifest) == {"val": ("seq_yaml",)}
    assert load_topic_map_payload(topic_map)["sequence_id"] == "seq_yaml"
