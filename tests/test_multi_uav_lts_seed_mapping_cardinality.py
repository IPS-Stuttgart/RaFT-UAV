from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.fixed_population import postprocess_fixed_population


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_seed_mapping_maximizes_valid_matches_before_iou(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"

    # The first-frame IoU matrix is approximately
    # [[1.000, 0.583], [0.526, 0.474]].  At a 0.5 gate, maximizing raw IoU
    # chooses the diagonal and then discards its second pair.  The crossed
    # assignment is the maximum-cardinality valid matching and preserves both
    # seeded trajectories.
    _write(
        labels / "S.txt",
        "1,7,0,0,10,10,1,1,1\n1,9,-9,0,19,10,1,1,1\n",
    )
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "1,2,-2,0,9,10,1,1,1\n"
        "2,1,1,0,10,10,0.8,1,1\n"
        "2,2,-1,0,9,10,0.8,1,1\n",
    )

    summary = postprocess_fixed_population(predictions, labels, output)

    assert summary.mapped_input_tracks == 2
    assert summary.dropped_input_tracks == 0
    assert summary.inserted_seed_rows == 0
    assert (output / "S.txt").read_text(encoding="utf-8").splitlines() == [
        "1,7,0,0,10,10,1,1,1",
        "1,9,-9,0,19,10,1,1,1",
        "2,7,-1,0,9,10,0.8,1,1",
        "2,9,1,0,10,10,0.8,1,1",
    ]
