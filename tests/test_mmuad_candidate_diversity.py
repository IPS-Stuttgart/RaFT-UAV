from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [1.0] * 4,
            "source": ["a", "a", "b", "c"],
            "track_id": ["best", "duplicate", "protected", "far"],
            "x_m": [0.0, 0.1, 0.2, 5.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.9, 0.1, 0.5],
            "candidate_reservoir_protected": [False, False, True, False],
        }
    )


def _uncertainty_rows(
    *,
    second_x_m: float,
    first_sigma_m: float,
    second_sigma_m: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 1.0],
            "source": ["a", "b"],
            "track_id": ["first", "second"],
            "x_m": [0.0, second_x_m],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.9],
            "predicted_sigma_m_hgb": [first_sigma_m, second_sigma_m],
        }
    )


def test_diversity_suppresses_near_duplicate_and_keeps_far_candidate() -> None:
    output = diversify_candidate_reservoir(_rows(), radius_m=1.0)
    assert set(output["track_id"]) == {"best", "protected", "far"}
    assert "duplicate" not in set(output["track_id"])


def test_diversity_can_disable_protected_override() -> None:
    output = diversify_candidate_reservoir(
        _rows(), radius_m=1.0, preserve_protected=False
    )
    assert set(output["track_id"]) == {"best", "far"}


def test_diversity_respects_per_frame_cap() -> None:
    output = diversify_candidate_reservoir(_rows(), radius_m=0.0, max_candidates_per_frame=2)
    assert len(output) == 2
    assert output["candidate_diversity_rank"].tolist() == [1, 2]


def test_diversity_does_not_expand_duplicate_input_index_labels() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 3,
            "time_s": [1.0] * 3,
            "track_id": ["best", "duplicate", "far"],
            "x_m": [0.0, 0.1, 5.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.9, 0.5],
            "candidate_reservoir_protected": [False, False, False],
        },
        index=[7, 7, 8],
    )

    output = diversify_candidate_reservoir(
        rows,
        radius_m=1.0,
        max_candidates_per_frame=2,
    )

    assert output["track_id"].tolist() == ["best", "far"]
    assert len(output) == 2


def test_uncertain_selected_candidate_preserves_nearby_alternative() -> None:
    rows = _uncertainty_rows(
        second_x_m=0.5,
        first_sigma_m=100.0,
        second_sigma_m=1.0,
    )

    fixed = diversify_candidate_reservoir(rows, radius_m=1.0)
    adaptive = diversify_candidate_reservoir(
        rows,
        radius_m=1.0,
        uncertainty_column="predicted_sigma_m_hgb",
        uncertainty_reference_m=10.0,
        uncertainty_exponent=1.0,
        min_radius_scale=0.1,
        max_radius_scale=10.0,
    )

    assert fixed["track_id"].tolist() == ["first"]
    assert adaptive["track_id"].tolist() == ["first", "second"]
    first = adaptive.loc[adaptive["track_id"] == "first"].iloc[0]
    assert first["candidate_diversity_effective_radius_m"] == pytest.approx(0.1)
    assert not bool(first["candidate_diversity_uncertainty_imputed"])


def test_precise_selected_candidate_suppresses_wider_duplicate_region() -> None:
    rows = _uncertainty_rows(
        second_x_m=2.0,
        first_sigma_m=1.0,
        second_sigma_m=100.0,
    )

    fixed = diversify_candidate_reservoir(rows, radius_m=1.0)
    adaptive = diversify_candidate_reservoir(
        rows,
        radius_m=1.0,
        uncertainty_column="predicted_sigma_m_hgb",
        uncertainty_reference_m=10.0,
        uncertainty_exponent=1.0,
        min_radius_scale=0.1,
        max_radius_scale=4.0,
    )

    assert fixed["track_id"].tolist() == ["first", "second"]
    assert adaptive["track_id"].tolist() == ["first"]
    assert adaptive["candidate_diversity_effective_radius_m"].iloc[0] == pytest.approx(4.0)


def test_uncertainty_diversity_imputes_invalid_sigma() -> None:
    rows = _uncertainty_rows(
        second_x_m=2.0,
        first_sigma_m=float("nan"),
        second_sigma_m=10.0,
    )

    output = diversify_candidate_reservoir(
        rows,
        radius_m=1.0,
        uncertainty_column="predicted_sigma_m_hgb",
        uncertainty_reference_m=10.0,
    )

    first = output.loc[output["track_id"] == "first"].iloc[0]
    assert first["candidate_diversity_uncertainty_m"] == pytest.approx(10.0)
    assert bool(first["candidate_diversity_uncertainty_imputed"])
    assert first["candidate_diversity_effective_radius_m"] == pytest.approx(1.0)


def test_uncertainty_diversity_rejects_missing_column_and_invalid_controls() -> None:
    with pytest.raises(ValueError, match="missing uncertainty column"):
        diversify_candidate_reservoir(
            _rows(),
            uncertainty_column="predicted_sigma_m_hgb",
        )
    with pytest.raises(ValueError, match="must not exceed"):
        diversify_candidate_reservoir(
            _uncertainty_rows(
                second_x_m=2.0,
                first_sigma_m=1.0,
                second_sigma_m=1.0,
            ),
            uncertainty_column="predicted_sigma_m_hgb",
            min_radius_scale=2.0,
            max_radius_scale=1.0,
        )
