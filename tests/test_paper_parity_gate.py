from __future__ import annotations

import json

import pandas as pd

from raft_uav.diagnostics.paper_parity_gate import (
    DEFAULT_REQUIRED_COUNT_DELTA_COLUMNS,
    PaperParityGateConfig,
    evaluate_paper_parity_gate,
    main,
    select_best_candidate,
)


def _summary_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "rank": 1,
        "failed": False,
        "paper_parity_score": 0.0,
        "count_abs_delta_total": 0,
        "kf_all_steps_mean_abs_delta_m": 1.25,
    }
    row.update({column: 0 for column in DEFAULT_REQUIRED_COUNT_DELTA_COLUMNS})
    row.update(overrides)
    return row


def test_gate_passes_exact_count_match() -> None:
    result = evaluate_paper_parity_gate(pd.DataFrame([_summary_row()]))

    assert result.passed
    assert result.reasons == ()
    assert result.best_rank == 1


def test_gate_fails_on_total_count_delta() -> None:
    result = evaluate_paper_parity_gate(
        pd.DataFrame(
            [
                _summary_row(
                    count_abs_delta_total=3,
                    radar_after_nis_count_delta=3,
                )
            ]
        )
    )

    assert not result.passed
    assert any("count_abs_delta_total" in reason for reason in result.reasons)
    assert any("radar_after_nis_count_delta" in reason for reason in result.reasons)


def test_gate_fails_when_best_candidate_failed() -> None:
    result = evaluate_paper_parity_gate(
        pd.DataFrame([_summary_row(failed=True, error="missing range_m")])
    )

    assert not result.passed
    assert any("marked failed" in reason for reason in result.reasons)
    assert any("missing range_m" in reason for reason in result.reasons)


def test_gate_can_skip_kf_mean_check_for_count_only_ci() -> None:
    frame = pd.DataFrame([_summary_row()]).drop(columns=["kf_all_steps_mean_abs_delta_m"])

    result = evaluate_paper_parity_gate(
        frame,
        config=PaperParityGateConfig(max_kf_mean_abs_delta_m=None),
    )

    assert result.passed


def test_gate_uses_rank_column_to_select_best_candidate() -> None:
    frame = pd.DataFrame(
        [
            _summary_row(rank=2, count_abs_delta_total=10, rf_raw_count_delta=10),
            _summary_row(rank=1, count_abs_delta_total=0, rf_raw_count_delta=0),
        ]
    )

    best = select_best_candidate(frame)
    result = evaluate_paper_parity_gate(frame)

    assert int(best["rank"]) == 1
    assert result.passed
    assert result.best_rank == 1


def test_gate_main_writes_json_payload(tmp_path) -> None:
    summary_csv = tmp_path / "paper_parity_grid_summary.csv"
    output_json = tmp_path / "gate.json"
    pd.DataFrame([_summary_row()]).to_csv(summary_csv, index=False)

    exit_code = main([str(summary_csv), "--output-json", str(output_json), "--quiet"])

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["best_rank"] == 1
