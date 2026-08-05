from __future__ import annotations

import numpy as np
import pandas as pd

from pyrecest.filters.candidate_mixture import GaussianMixtureMeasurementFactor
import raft_uav.mmuad._candidate_mixture_pyrecest as pyrecest_adapter
import raft_uav.mmuad.candidate_mixture_map as candidate_mixture_map


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "near",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.8,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "dynamic",
                "track_id": "middle",
                "candidate_branch": "dynamic",
                "x_m": 3.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.5,
                "predicted_sigma_m": 2.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "dynamic",
                "track_id": "far",
                "candidate_branch": "dynamic",
                "x_m": 0.0,
                "y_m": 4.0,
                "z_m": 0.0,
                "ranker_score": 0.2,
                "predicted_sigma_m": 4.0,
            },
        ]
    )


def test_candidate_mixture_delegates_factor_evaluation_to_pyrecest(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def recording_factor(*args, **kwargs):
        calls.append(dict(kwargs))
        return GaussianMixtureMeasurementFactor(*args, **kwargs)

    monkeypatch.setattr(pyrecest_adapter, "_load_factor_type", lambda: recording_factor)
    result = candidate_mixture_map.compute_candidate_responsibilities(
        _candidate_rows(),
        np.zeros(3),
        config=candidate_mixture_map.CandidateMixtureMapConfig(
            top_k=0,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            sigma_log_weight=3.0,
        ),
    )

    assert candidate_mixture_map._mixture_response.__module__ == pyrecest_adapter.__name__
    assert len(calls) == 1
    assert np.asarray(calls[0]["means"]).shape == (3, 3)
    assert np.asarray(calls[0]["covariances"]).shape == (3, 3, 3)
    assert calls[0]["log_determinant_weight"] == 1.0
    assert np.isclose(result["mixture_responsibility"].sum(), 1.0)


def test_pyrecest_adapter_matches_legacy_isotropic_equations() -> None:
    config = candidate_mixture_map.CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        sigma_column="predicted_sigma_m",
        score_normalization="none",
        score_weight=0.7,
        temperature=1.3,
        sigma_log_weight=2.5,
        loss="huber",
        huber_delta=0.8,
        smoothness_weight=0.0,
        iterations=1,
    )
    rows = candidate_mixture_map.normalize_candidate_columns(_candidate_rows())
    rows = rows.copy().reset_index(drop=True)
    rows["_mixture_input_row"] = np.arange(len(rows), dtype=int)
    frames = candidate_mixture_map._prepare_candidate_frames(rows, config=config)
    state = np.asarray([[1.0, 0.5, 0.0]])

    response = candidate_mixture_map._mixture_response(
        frames,
        state,
        config=config,
    )[0]
    frame = frames[0]
    positions = np.asarray(frame["positions"], dtype=float)
    sigmas = np.asarray(frame["sigmas"], dtype=float)
    scores = np.asarray(frame["normalized_scores"], dtype=float)
    distances = np.linalg.norm(positions - state[0], axis=1)
    normalized_residual = distances / sigmas
    robust_cost = np.where(
        normalized_residual <= config.huber_delta,
        0.5 * normalized_residual**2,
        config.huber_delta * (normalized_residual - 0.5 * config.huber_delta),
    )
    log_weight = (
        config.score_weight * scores / config.temperature
        - robust_cost
        - config.sigma_log_weight * np.log(sigmas)
    )
    shifted = log_weight - np.max(log_weight)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    pseudo_position = weights @ positions
    spread_variance = np.sum(
        weights * np.sum((positions - pseudo_position) ** 2, axis=1) / 3.0
    )
    effective_variance = np.sum(weights * sigmas**2) + spread_variance

    np.testing.assert_allclose(response["distances"], distances)
    np.testing.assert_allclose(response["normalized_residual"], normalized_residual)
    np.testing.assert_allclose(response["robust_cost"], robust_cost)
    np.testing.assert_allclose(response["log_weight"], log_weight)
    np.testing.assert_allclose(response["weights"], weights)
    np.testing.assert_allclose(response["pseudo_position"], pseudo_position)
    np.testing.assert_allclose(
        response["effective_sigma_m"],
        np.sqrt(effective_variance),
    )
    np.testing.assert_allclose(
        response["measurement_precision"],
        1.0 / effective_variance,
    )
