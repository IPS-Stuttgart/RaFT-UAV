from __future__ import annotations

import csv
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.scenario_expert_gate import (
    CandidateScore,
    GateConfig,
    fit_guarded_policy,
    fit_prefix_policy,
    load_score_bank,
    materialize_policy,
)


def _score(sequence: str, candidate: str, hota: float) -> CandidateScore:
    return CandidateScore(
        sequence=sequence,
        candidate=candidate,
        hota=hota,
        mota=0.8,
        idf1=0.8,
    )


def _write_score_csv(path: Path, rows: dict[str, CandidateScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sequence", "hota_at_005", "mota", "idf1"),
        )
        writer.writeheader()
        for sequence in sorted(rows):
            row = rows[sequence]
            writer.writerow(
                {
                    "sequence": sequence,
                    "hota_at_005": row.hota,
                    "mota": row.mota,
                    "idf1": row.idf1,
                }
            )


def test_load_score_bank_accepts_metrics_sequence_csv_hota_column(
    tmp_path: Path,
) -> None:
    raw = {"C_00": _score("C_00", "raw", 0.8)}
    graph = {"C_00": _score("C_00", "graph", 0.9)}
    raw_path = tmp_path / "raw.csv"
    graph_path = tmp_path / "graph.csv"
    _write_score_csv(raw_path, raw)
    _write_score_csv(graph_path, graph)

    bank = load_score_bank(
        {"raw": raw_path, "graph": graph_path},
        raw_candidate="raw",
    )

    assert bank["graph"]["C_00"].hota == pytest.approx(0.9)


def test_prefix_selection_rejects_uncertain_gain_by_familywise_bound() -> None:
    sequences = tuple(f"C_{index:02d}" for index in range(6))
    raw = {sequence: _score(sequence, "raw", 0.8) for sequence in sequences}
    deltas = (0.10, -0.08, 0.10, -0.08, 0.10, -0.08)
    graph = {
        sequence: _score(sequence, "graph", 0.8 + delta)
        for sequence, delta in zip(sequences, deltas, strict=True)
    }
    config = GateConfig(
        fold_count=3,
        min_prefix_samples=2,
        min_train_hota_gain=0.0,
        prior_strength=0.0,
        min_cv_hota_gain=0.0,
        bootstrap_samples=5000,
        familywise_alpha=0.05,
        min_train_hota_ci_low=0.0,
    )

    mapping, diagnostics = fit_prefix_policy(
        {"raw": raw, "graph": graph},
        sequences,
        config,
    )

    assert mapping == {"C": "raw"}
    graph_row = next(
        row
        for row in diagnostics["C"]["candidates"]
        if row["candidate"] == "graph"
    )
    assert graph_row["mean_hota_gain"] > 0.0
    assert graph_row["hota_gain_ci_low"] < 0.0
    assert not graph_row["eligible"]


def test_guarded_policy_records_positive_paired_cv_bound() -> None:
    raw: dict[str, CandidateScore] = {}
    graph: dict[str, CandidateScore] = {}
    for prefix in ("C", "T"):
        for index in range(6):
            sequence = f"{prefix}_{index:02d}"
            raw[sequence] = _score(sequence, "raw", 0.8)
            graph[sequence] = _score(
                sequence,
                "graph",
                0.83 if prefix == "C" else 0.78,
            )
    config = GateConfig(
        fold_count=3,
        seed=7,
        min_prefix_samples=2,
        min_train_hota_gain=0.005,
        prior_strength=0.0,
        min_cv_hota_gain=0.005,
        bootstrap_samples=5000,
        min_train_hota_ci_low=0.0,
        min_cv_hota_ci_low=0.0,
    )

    policy, _rows = fit_guarded_policy({"raw": raw, "graph": graph}, config)

    summary = policy["cross_validation"]
    assert not policy["raw_fallback"]
    assert summary["paired_hota_gain_ci_low"] >= 0.0
    assert summary["paired_hota_gain_ci_high"] > 0.0


def _write_predictions(path: Path, values: dict[str, bytes]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, payload in values.items():
        (path / name).write_bytes(payload)


def _policy(mapping: dict[str, str]) -> dict[str, object]:
    return {
        "schema": "raft-uav-multi-uav-lts-scenario-expert-policy-v1",
        "raw_candidate": "raw",
        "prefix_to_candidate": mapping,
        "raw_fallback": False,
    }


def test_failed_materialization_preserves_existing_output(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    graph = tmp_path / "graph"
    output = tmp_path / "output"
    _write_predictions(raw, {"C_00.txt": b"raw-c\n", "T_00.txt": b"raw-t\n"})
    _write_predictions(graph, {"C_00.txt": b"graph-c\n"})
    _write_predictions(output, {"sentinel.txt": b"keep\n"})

    with pytest.raises(ValueError, match="coverage mismatch"):
        materialize_policy(
            _policy({"C": "graph", "T": "raw"}),
            {"raw": raw, "graph": graph},
            output,
        )

    assert sorted(path.name for path in output.iterdir()) == ["sentinel.txt"]
    assert (output / "sentinel.txt").read_bytes() == b"keep\n"


def test_materialization_rejects_output_ancestor_of_candidate(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw"
    _write_predictions(raw, {"C_00.txt": b"raw\n"})

    with pytest.raises(ValueError, match="disjoint"):
        materialize_policy(
            _policy({"C": "raw"}),
            {"raw": raw},
            tmp_path / "inputs",
        )


def test_gate_config_rejects_boolean_integer_controls() -> None:
    with pytest.raises(ValueError, match="fold_count"):
        GateConfig(fold_count=True).validate()
