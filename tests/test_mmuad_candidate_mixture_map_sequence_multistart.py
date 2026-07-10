from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad import candidate_mixture_map_sequence_multistart as sequence_multistart


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "a",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqB",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "b",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )


def _start_rows(marker: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "state_x_m": 0.0 if marker == "raw" else 10.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
                "start_marker": marker,
            },
            {
                "sequence_id": "seqB",
                "time_s": 0.0,
                "state_x_m": 0.0 if marker == "raw" else 10.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
                "start_marker": marker,
            },
        ]
    )


def _fake_core_result(candidates: pd.DataFrame, initial_estimates: pd.DataFrame | None):
    initial = pd.DataFrame() if initial_estimates is None else pd.DataFrame(initial_estimates)
    estimate_records = []
    assignment_records = []
    for sequence_id in sorted(candidates["sequence_id"].astype(str).unique()):
        sequence_initial = initial.loc[
            initial.get("sequence_id", pd.Series(dtype=str)).astype(str) == sequence_id
        ]
        marker = "core-default"
        if not sequence_initial.empty and "start_marker" in sequence_initial.columns:
            marker = str(sequence_initial.iloc[0]["start_marker"])
        preferred = "raw" if sequence_id == "seqA" else "translated"
        log_weight = 0.0 if marker == preferred else -5.0
        state_x = 0.0 if marker == "raw" else 10.0
        estimate_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": 0.0,
                "state_x_m": state_x,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
                "mixture_assignment_entropy": 0.0,
                "mixture_effective_sigma_m": 1.0,
            }
        )
        assignment_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": 0.0,
                "mixture_log_weight": log_weight,
            }
        )
    return core.CandidateMixtureMapResult(
        estimates=pd.DataFrame.from_records(estimate_records),
        assignments=pd.DataFrame.from_records(assignment_records),
        iteration_summary=pd.DataFrame(),
        summary={"metrics": {"pooled": {}, "sequences": {}}},
    )


def test_sequence_multistart_selects_different_restart_per_sequence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    starts = {
        "branch:raw": _start_rows("raw"),
        "branch:translated": _start_rows("translated"),
    }
    monkeypatch.setattr(
        sequence_multistart.multistart,
        "build_candidate_mixture_initializations",
        lambda *args, **kwargs: starts,
    )

    call_markers: list[dict[str, str]] = []

    def fake_run(
        candidates: pd.DataFrame,
        *,
        config,
        initial_estimates=None,
        truth=None,
        **kwargs,
    ):
        initial = pd.DataFrame() if initial_estimates is None else pd.DataFrame(initial_estimates)
        markers = {}
        if not initial.empty and "start_marker" in initial.columns:
            markers = {
                str(sequence_id): str(group.iloc[0]["start_marker"])
                for sequence_id, group in initial.groupby("sequence_id", sort=True)
            }
        call_markers.append(markers)
        return _fake_core_result(pd.DataFrame(candidates), initial_estimates)

    monkeypatch.setattr(sequence_multistart.core, "run_candidate_mixture_map", fake_run)

    result = sequence_multistart.run_sequence_multistart_candidate_mixture_map(
        _candidate_rows(),
        mixture_config=core.CandidateMixtureMapConfig(
            top_k=0,
            smoothness_weight=0.0,
            anchor_weight=0.0,
        ),
    )

    assert result.selected_start == "per-sequence"
    assert result.selected_start_by_sequence == {
        "seqA": "branch:raw",
        "seqB": "branch:translated",
    }
    selected = result.start_summary.loc[result.start_summary["selected"]]
    assert selected.set_index("sequence_id")["start_name"].to_dict() == {
        "seqA": "branch:raw",
        "seqB": "branch:translated",
    }
    assert result.selected_initializations.set_index("sequence_id")["start_marker"].to_dict() == {
        "seqA": "raw",
        "seqB": "translated",
    }
    assert call_markers[-1] == {"seqA": "raw", "seqB": "translated"}
    assert result.selected_result.assignments["mixture_log_weight"].tolist() == [0.0, 0.0]
    assert result.summary["truth_used_for_selection"] is False
    assert result.summary["final_result_reused"] is False

    paths = sequence_multistart.write_sequence_multistart_candidate_mixture_outputs(
        result,
        output_dir=tmp_path,
    )
    assert paths["sequence_multistart_summary_csv"].exists()
    assert paths["selected_initializations_csv"].exists()
    summary = json.loads(paths["sequence_multistart_summary_json"].read_text(encoding="utf-8"))
    assert summary["selected_start_by_sequence"] == {
        "seqA": "branch:raw",
        "seqB": "branch:translated",
    }


def test_sequence_multistart_reuses_result_when_all_sequences_choose_same_start(
    monkeypatch,
) -> None:
    starts = {"branch:raw": _start_rows("raw")}
    monkeypatch.setattr(
        sequence_multistart.multistart,
        "build_candidate_mixture_initializations",
        lambda *args, **kwargs: starts,
    )
    call_count = 0

    def fake_run(
        candidates: pd.DataFrame,
        *,
        config,
        initial_estimates=None,
        truth=None,
        **kwargs,
    ):
        nonlocal call_count
        call_count += 1
        return _fake_core_result(pd.DataFrame(candidates), initial_estimates)

    monkeypatch.setattr(sequence_multistart.core, "run_candidate_mixture_map", fake_run)

    result = sequence_multistart.run_sequence_multistart_candidate_mixture_map(
        _candidate_rows(),
        mixture_config=core.CandidateMixtureMapConfig(
            top_k=0,
            smoothness_weight=0.0,
            anchor_weight=0.0,
        ),
    )

    assert result.selected_start == "branch:raw"
    assert result.summary["final_result_reused"] is True
    assert call_count == 1
