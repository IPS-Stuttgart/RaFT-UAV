from __future__ import annotations

import csv
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.scenario_expert_gate import (
    CandidateScore,
    GateConfig,
    fit_guarded_policy,
    load_score_bank,
    materialize_policy,
    sequence_prefix,
)


def _score(
    sequence: str,
    candidate: str,
    *,
    hota: float,
    mota: float,
    idf1: float,
) -> CandidateScore:
    return CandidateScore(
        sequence=sequence,
        candidate=candidate,
        hota=hota,
        mota=mota,
        idf1=idf1,
    )


def _bank() -> dict[str, dict[str, CandidateScore]]:
    raw: dict[str, CandidateScore] = {}
    graph: dict[str, CandidateScore] = {}
    tube: dict[str, CandidateScore] = {}
    for prefix in ("C", "T"):
        for index in range(6):
            sequence = f"{prefix}_{index:02d}"
            raw[sequence] = _score(
                sequence,
                "raw",
                hota=0.80,
                mota=0.70,
                idf1=0.75,
            )
            if prefix == "C":
                graph[sequence] = _score(
                    sequence,
                    "graph",
                    hota=0.83,
                    mota=0.701,
                    idf1=0.76,
                )
            else:
                graph[sequence] = _score(
                    sequence,
                    "graph",
                    hota=0.78,
                    mota=0.68,
                    idf1=0.72,
                )
            tube[sequence] = _score(
                sequence,
                "tube",
                hota=0.801,
                mota=0.699,
                idf1=0.749,
            )
    return {"raw": raw, "graph": graph, "tube": tube}


def _config(**updates: object) -> GateConfig:
    values: dict[str, object] = {
        "fold_count": 3,
        "seed": 7,
        "min_prefix_samples": 2,
        "min_train_hota_gain": 0.005,
        "prior_strength": 0.0,
        "max_train_mota_drop": 0.002,
        "max_train_idf1_drop": 0.002,
        "min_cv_hota_gain": 0.005,
        "max_cv_mota_drop": 0.002,
        "max_cv_idf1_drop": 0.002,
        "max_worst_prefix_hota_drop": 0.001,
    }
    values.update(updates)
    return GateConfig(**values)


def test_sequence_prefix_removes_only_numeric_suffix() -> None:
    assert sequence_prefix("BB2P_03") == "BB2P"
    assert sequence_prefix("Takeoff_00.txt") == "Takeoff"
    assert sequence_prefix("custom_scene") == "custom_scene"


def test_guarded_policy_selects_graph_only_for_supported_prefix() -> None:
    policy, cv_rows = fit_guarded_policy(_bank(), _config())

    assert not policy["raw_fallback"]
    assert policy["prefix_to_candidate"] == {"C": "graph", "T": "raw"}
    assert policy["cross_validation"]["passed"]
    assert policy["cross_validation"]["mean_hota_delta"] == pytest.approx(0.015)
    assert {row["candidate"] for row in cv_rows if row["prefix"] == "C"} == {
        "graph"
    }
    assert {row["candidate"] for row in cv_rows if row["prefix"] == "T"} == {
        "raw"
    }


def test_guarded_policy_falls_back_to_raw_when_cv_gain_is_too_small() -> None:
    policy, _rows = fit_guarded_policy(
        _bank(),
        _config(min_cv_hota_gain=0.02),
    )

    assert policy["raw_fallback"]
    assert policy["prefix_to_candidate"] == {"C": "raw", "T": "raw"}
    assert policy["cross_validation"]["rejection_reasons"] == [
        "mean_cv_hota_gain"
    ]


def _write_score_csv(
    path: Path,
    candidate: str,
    rows: dict[str, CandidateScore],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sequence",
                "CODABENCH_HOTA",
                "CODABENCH_MOTA",
                "CODABENCH_IDF1",
            ),
        )
        writer.writeheader()
        for sequence in sorted(rows):
            score = rows[sequence]
            assert score.candidate == candidate
            writer.writerow(
                {
                    "sequence": sequence,
                    "CODABENCH_HOTA": score.hota,
                    "CODABENCH_MOTA": score.mota,
                    "CODABENCH_IDF1": score.idf1,
                }
            )


def test_score_bank_requires_exact_sequence_coverage(tmp_path: Path) -> None:
    bank = _bank()
    raw_path = tmp_path / "raw.csv"
    graph_path = tmp_path / "graph.csv"
    _write_score_csv(raw_path, "raw", bank["raw"])
    incomplete = dict(bank["graph"])
    incomplete.pop("C_00")
    _write_score_csv(graph_path, "graph", incomplete)

    with pytest.raises(ValueError, match="coverage mismatch"):
        load_score_bank(
            {"raw": raw_path, "graph": graph_path},
            raw_candidate="raw",
        )


def _write_prediction_set(
    root: Path,
    values: dict[str, bytes],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in values.items():
        (root / name).write_bytes(payload)


def test_materialization_preserves_selected_candidate_bytes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    graph_dir = tmp_path / "graph"
    output_dir = tmp_path / "selected"
    _write_prediction_set(
        raw_dir,
        {
            "C_00.txt": b"raw-cloud\n",
            "T_00.txt": b"raw-tree\n",
        },
    )
    _write_prediction_set(
        graph_dir,
        {
            "C_00.txt": b"graph-cloud\n",
            "T_00.txt": b"graph-tree\n",
        },
    )
    policy = {
        "schema": "raft-uav-multi-uav-lts-scenario-expert-policy-v1",
        "raw_candidate": "raw",
        "prefix_to_candidate": {"C": "graph", "T": "raw"},
        "raw_fallback": False,
    }

    summary = materialize_policy(
        policy,
        {"raw": raw_dir, "graph": graph_dir},
        output_dir,
    )

    assert (output_dir / "C_00.txt").read_bytes() == b"graph-cloud\n"
    assert (output_dir / "T_00.txt").read_bytes() == b"raw-tree\n"
    assert summary["file_count"] == 2
    assert summary["selected_candidate_counts"] == {"graph": 1, "raw": 1}


def test_materialization_rejects_output_inside_candidate(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_prediction_set(raw_dir, {"C_00.txt": b"raw\n"})
    policy = {
        "schema": "raft-uav-multi-uav-lts-scenario-expert-policy-v1",
        "raw_candidate": "raw",
        "prefix_to_candidate": {"C": "raw"},
        "raw_fallback": True,
    }

    with pytest.raises(ValueError, match="must not equal or be inside"):
        materialize_policy(
            policy,
            {"raw": raw_dir},
            raw_dir / "selected",
        )
