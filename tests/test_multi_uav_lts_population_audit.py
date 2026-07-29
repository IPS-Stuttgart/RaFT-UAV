from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.population_audit import audit_first_frame_population


def test_population_audit_finds_late_births_and_reappearances(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    truth.mkdir()
    truth.joinpath("S.txt").write_text(
        "1,1,0,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,1,1,1\n"
        "4,1,3,0,10,10,1,1,1\n"
        "3,2,5,0,10,10,1,1,1\n",
        encoding="utf-8",
    )

    audit = audit_first_frame_population(truth)

    assert audit.identity_count == 2
    assert audit.frame_one_identity_count == 1
    assert audit.late_birth_identity_count == 1
    assert audit.late_birth_fraction == 0.5
    assert audit.reappearing_identity_count == 1
    assert audit.maximum_gap_frames == 1
    assert audit.sequences[0].late_birth_ids == (2,)
    assert audit.sequences[0].reappearing_ids == (1,)
