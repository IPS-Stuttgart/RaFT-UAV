from __future__ import annotations

from raft_uav.multi_uav_lts.scenario_box_policy import (
    assemble_scenario_policy_predictions,
)


def _write(path, x: float) -> None:
    path.write_text(f"1,1,{x},10,10,10,1,1,1\n", encoding="utf-8")


def test_policy_selects_variant_per_prefix_from_complementary_fold(tmp_path) -> None:
    truth = tmp_path / "truth"
    reference_a = tmp_path / "reference_a"
    reference_b = tmp_path / "reference_b"
    target_a = tmp_path / "target_a"
    target_b = tmp_path / "target_b"
    output = tmp_path / "output"
    for path in (truth, reference_a, reference_b, target_a, target_b):
        path.mkdir()

    sequences = ("A_00", "A_01", "B_00", "B_01")
    for sequence in sequences:
        _write(truth / f"{sequence}.txt", 10.0)
        # Reference candidate a is exact for A and a complete miss for B.
        _write(reference_a / f"{sequence}.txt", 10.0 if sequence.startswith("A_") else 100.0)
        # Reference candidate b has the complementary scenario behavior.
        _write(reference_b / f"{sequence}.txt", 100.0 if sequence.startswith("A_") else 10.0)
        # Target contents are deliberately distinct so routing is directly observable.
        _write(target_a / f"{sequence}.txt", 20.0)
        _write(target_b / f"{sequence}.txt", 30.0)

    summary = assemble_scenario_policy_predictions(
        {"a": reference_a, "b": reference_b},
        {"a": target_a, "b": target_b},
        truth,
        (("A_00", "B_00"), ("A_01", "B_01")),
        output,
    )

    assert summary.sequence_count == 4
    selected = {(row.fold, row.prefix): row.selected_candidate for row in summary.selections}
    assert selected == {(0, "A"): "a", (0, "B"): "b", (1, "A"): "a", (1, "B"): "b"}
    assert all(not row.used_global_fallback for row in summary.selections)
    assert (output / "A_00.txt").read_text(encoding="utf-8").split(",")[2] == "20.0"
    assert (output / "B_00.txt").read_text(encoding="utf-8").split(",")[2] == "30.0"


def test_policy_falls_back_to_global_training_panel_for_unseen_prefix(tmp_path) -> None:
    truth = tmp_path / "truth"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    truth.mkdir()
    candidate.mkdir()
    for sequence in ("A_00", "B_00"):
        _write(truth / f"{sequence}.txt", 10.0)
        _write(candidate / f"{sequence}.txt", 10.0)

    summary = assemble_scenario_policy_predictions(
        {"only": candidate},
        {"only": candidate},
        truth,
        (("A_00",), ("B_00",)),
        output,
    )

    assert all(row.used_global_fallback for row in summary.selections)
    assert {row.selected_candidate for row in summary.selections} == {"only"}
