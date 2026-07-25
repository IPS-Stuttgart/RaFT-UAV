from __future__ import annotations

from pathlib import Path

from raft_uav.mmuad.layout import inspect_mmuad_layout


def test_layout_keeps_unrelated_token_prefixes_as_sequence_names(
    tmp_path: Path,
) -> None:
    for sequence_name in ("radarized", "trainwreck"):
        sequence = tmp_path / sequence_name
        sequence.mkdir()
        (sequence / "candidates.csv").write_text(
            "time_s,x_m,y_m,z_m\n",
            encoding="utf-8",
        )

    summary = inspect_mmuad_layout(tmp_path)

    assert [
        row["sequence_id"] for row in summary["sequence_candidates"]
    ] == ["radarized", "trainwreck"]


def test_layout_preserves_compact_modality_suffix_matching(
    tmp_path: Path,
) -> None:
    modality = tmp_path / "val" / "seq_compact" / "LivoxAvia2"
    modality.mkdir(parents=True)
    (modality / "points.npy").write_bytes(b"placeholder")

    summary = inspect_mmuad_layout(tmp_path)

    assert [
        row["sequence_id"] for row in summary["sequence_candidates"]
    ] == ["seq_compact"]
