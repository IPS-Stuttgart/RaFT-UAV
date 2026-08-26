from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_grid import run_candidate_reservoir_offset_grid


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-a"],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "livox_avia"],
            "track_id": ["good", "bad"],
            "candidate_branch": ["raw", "translated"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "ranker_score": [0.9, 0.1],
            "confidence": [0.9, 0.1],
        }
    )


def _truth_rows(sequence_id: str = "seq-a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [sequence_id],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("global_top_n", True),
        ("global_top_n", 1.5),
        ("per_source_top_n", -1),
        ("per_branch_top_n", np.array([1])),
        ("max_candidates_per_frame", -1),
    ],
)
def test_reservoir_grid_rejects_lossy_integer_controls(name: str, value: object) -> None:
    kwargs = {name: value}
    with pytest.raises(ValueError, match=rf"{name} must be a non-negative integer scalar"):
        run_candidate_reservoir_offset_grid(_candidate_rows(), **kwargs)


@pytest.mark.parametrize(
    "top_k_values",
    [
        (),
        (True,),
        (1.5,),
        (0,),
        (np.array([1]),),
        "1",
    ],
)
def test_reservoir_grid_rejects_invalid_top_k_values(top_k_values: object) -> None:
    with pytest.raises(ValueError, match="top_k_values"):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            top_k_values=top_k_values,
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("score_floor_quantile", True),
        ("score_floor_quantile", float("nan")),
        ("score_floor_quantile", -0.01),
        ("score_floor_quantile", 1.01),
        ("cap_reason_bonus", False),
        ("cap_reason_bonus", float("inf")),
        ("max_truth_time_delta_s", True),
        ("max_truth_time_delta_s", float("nan")),
        ("max_truth_time_delta_s", -0.01),
    ],
)
def test_reservoir_grid_rejects_invalid_float_controls(name: str, value: object) -> None:
    kwargs = {name: value}
    with pytest.raises(ValueError):
        run_candidate_reservoir_offset_grid(_candidate_rows(), **kwargs)


def test_reservoir_grid_rejects_non_boolean_write_best_control() -> None:
    with pytest.raises(ValueError, match="write_best_reservoir must be a Boolean scalar"):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            write_best_reservoir="False",
        )


def test_reservoir_grid_rejects_colliding_normalized_offset_names() -> None:
    with pytest.raises(ValueError, match="must remain unique after filename normalization"):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            branch_offset_grid=["raw/a=0,1", "raw a=0,1"],
        )


def test_reservoir_grid_rejects_reserved_identity_offset_name() -> None:
    with pytest.raises(ValueError, match="'__none__' is reserved"):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            branch_offset_grid=["__none__=0,1"],
        )


def test_reservoir_grid_rejects_colliding_serialized_configurations() -> None:
    with pytest.raises(ValueError, match="configurations collide after label formatting"):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            branch_offset_grid=[
                "a=0,1",
                "b=0,2",
                "a_1__branch_b=0,2",
            ],
        )


def test_reservoir_grid_accepts_zero_dimensional_scalar_controls() -> None:
    summary, best = run_candidate_reservoir_offset_grid(
        _candidate_rows(),
        truth=_truth_rows(),
        global_top_n=np.array(1),
        per_source_top_n=np.array(0),
        per_branch_top_n=np.array(0),
        max_candidates_per_frame=np.array(1),
        score_floor_quantile=np.array(0.0),
        cap_reason_bonus=np.float64(0.25),
        top_k_values=(np.array(1),),
        max_truth_time_delta_s=np.array(0.0),
        selection_metric="oracle_top1_3d_m_mse",
        write_best_reservoir=np.bool_(True),
    )

    assert summary.iloc[0]["oracle_top1_3d_m_mse"] == 0.0
    assert best is not None
    assert best["track_id"].tolist() == ["good"]


def test_reservoir_grid_allows_unscored_truth_without_best_selection() -> None:
    summary, best = run_candidate_reservoir_offset_grid(
        _candidate_rows(),
        truth=_truth_rows("other-sequence"),
        global_top_n=1,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=1,
        top_k_values=(1,),
        selection_metric="oracle_top1_3d_m_mse",
        write_best_reservoir=False,
    )

    assert not summary.empty
    assert best is None


def test_reservoir_grid_rejects_unscored_truth_before_output_creation(tmp_path) -> None:
    output_dir = tmp_path / "grid"

    with pytest.raises(ValueError, match="was not produced|has no finite values"):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            truth=_truth_rows("other-sequence"),
            output_dir=output_dir,
            global_top_n=1,
            per_source_top_n=0,
            per_branch_top_n=0,
            max_candidates_per_frame=1,
            top_k_values=(1,),
            selection_metric="oracle_top1_3d_m_mse",
            write_best_reservoir=True,
        )

    assert not output_dir.exists()


def test_reservoir_grid_rejects_selection_metric_missing_from_requested_top_k() -> None:
    with pytest.raises(ValueError, match="selection metric .* was not produced"):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            truth=_truth_rows(),
            top_k_values=(1,),
            selection_metric="oracle_top5_3d_m_mse",
            write_best_reservoir=True,
        )
