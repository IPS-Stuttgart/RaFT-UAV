from __future__ import annotations

from pathlib import Path

from raft_uav.mmuad.layout import inspect_mmuad_layout


def test_layout_does_not_treat_gt_inside_an_unrelated_word_as_truth(
    tmp_path: Path,
) -> None:
    sequence = tmp_path / "seq_length"
    sequence.mkdir()
    (sequence / "length.csv").write_text("value\n1\n", encoding="utf-8")

    summary = inspect_mmuad_layout(tmp_path)

    assert summary["category_counts"] == {"table_other": 1}
    assert summary["sequence_candidates"] == [
        {
            "sequence_id": "seq_length",
            "file_count": 1,
            "categories": {"table_other": 1},
            "has_topic_map_export": False,
            "has_native_topic_map": False,
            "has_candidates_or_points": False,
            "has_truth_or_labels": False,
            "has_class_labels": False,
            "has_calibration": False,
        }
    ]


def test_layout_still_recognizes_gt_as_a_truth_marker(tmp_path: Path) -> None:
    sequence = tmp_path / "seq_truth"
    sequence.mkdir()
    (sequence / "gt.csv").write_text("time_s,x_m,y_m,z_m\n", encoding="utf-8")

    summary = inspect_mmuad_layout(tmp_path)

    assert summary["category_counts"] == {"truth_or_label": 1}
    assert summary["sequence_candidates"][0]["has_truth_or_labels"] is True
