"""One-shot patch for delegating RaFT-UAV candidate-mixture numerics to PyRecEst."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

TARGET = Path("src/raft_uav/mmuad/candidate_mixture_map.py")
EXPECTED_BLOB_SHA = "8fb16e5a30253c44fffcb7e562a69723c09565fb"
PYRECEST_IMPORT = (
    "from pyrecest.filters.candidate_mixture import "
    "GaussianMixtureMeasurementFactor\n"
)

NEW_MIXTURE_RESPONSE = '''def _mixture_response(
    frames: Sequence[dict[str, Any]],
    state: np.ndarray,
    *,
    config: CandidateMixtureMapConfig,
) -> list[dict[str, Any]]:
    response: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        positions = np.asarray(frame["positions"], dtype=float)
        sigmas = np.asarray(frame["sigmas"], dtype=float)
        scores = np.asarray(frame["normalized_scores"], dtype=float)
        measurement_dim = int(positions.shape[1])
        sigma_log_weight = float(config.sigma_log_weight)
        if sigma_log_weight == 0.0:
            covariances = np.eye(measurement_dim, dtype=float)
            log_determinant_weight = 0.0
        else:
            covariances = (
                sigmas[:, None, None] ** 2
                * np.eye(measurement_dim, dtype=float)[None, :, :]
            )
            log_determinant_weight = sigma_log_weight / float(measurement_dim)
        factor = GaussianMixtureMeasurementFactor(
            means=positions,
            covariances=covariances,
            log_weights=(
                float(config.score_weight) * scores / float(config.temperature)
            ),
            loss=config.loss,
            huber_delta=float(config.huber_delta),
            log_determinant_weight=log_determinant_weight,
        )
        evaluation = factor.evaluate(state[frame_index])
        weights = evaluation.responsibilities
        floor = float(config.uniform_weight_floor)
        if floor > 0.0:
            weights = (1.0 - floor) * weights + floor / len(weights)
        rows = frame["rows"]
        weights = _apply_label_balance(
            weights,
            rows["candidate_branch"].astype(str).to_numpy()
            if "candidate_branch" in rows
            else np.full(len(weights), "unknown", dtype=object),
            balance=float(config.branch_balance),
        )
        weights = _apply_label_balance(
            weights,
            rows["source"].astype(str).to_numpy()
            if "source" in rows
            else np.full(len(weights), "unknown", dtype=object),
            balance=float(config.source_balance),
        )
        floor = float(config.responsibility_floor)
        if floor > 0.0:
            weights = (1.0 - floor) * weights + floor / len(weights)
        weights = _normalize_probability(weights)
        pseudo, effective_covariance = factor.moment_match(weights)
        effective_variance = max(
            float(np.trace(effective_covariance)) / float(measurement_dim),
            1.0e-12,
        )
        precision = float(
            np.clip(
                1.0 / effective_variance,
                float(config.min_measurement_precision),
                float(config.max_measurement_precision),
            )
        )
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))
        response.append(
            {
                "weights": weights,
                "distances": np.linalg.norm(evaluation.residuals, axis=1),
                "normalized_residual": evaluation.mahalanobis_distances,
                "robust_cost": evaluation.robust_costs,
                "log_weight": evaluation.component_log_weights,
                "pseudo_position": pseudo,
                "effective_sigma_m": float(np.sqrt(effective_variance)),
                "measurement_precision": precision,
                "entropy": entropy,
                "effective_candidate_count": float(np.exp(entropy)),
                "dominant_index": int(np.argmax(weights)),
            }
        )
    return response


'''


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _replace_once(source: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}, found {count}")
    return updated


def main() -> None:
    original_bytes = TARGET.read_bytes()
    blob_sha = _git_blob_sha(original_bytes)
    if blob_sha != EXPECTED_BLOB_SHA:
        raise RuntimeError(
            f"unexpected {TARGET} blob {blob_sha}; expected {EXPECTED_BLOB_SHA}"
        )
    source = original_bytes.decode("utf-8")

    import_anchor = "import pandas as pd\n"
    if source.count(import_anchor) != 1:
        raise RuntimeError("unexpected pandas import count")
    if PYRECEST_IMPORT in source:
        raise RuntimeError("PyRecEst candidate-mixture import already present")
    source = source.replace(import_anchor, import_anchor + PYRECEST_IMPORT, 1)

    source = _replace_once(
        source,
        r"def _mixture_response\(.*?(?=def _solve_smooth_trajectory\()",
        NEW_MIXTURE_RESPONSE,
        "mixture-response function",
    )
    source = _replace_once(
        source,
        r"\ndef _robust_cost\(.*?(?=\ndef _stable_softmax\()",
        "\n",
        "local robust-cost helper",
    )
    source = _replace_once(
        source,
        r"\ndef _stable_softmax\(.*?(?=\ndef _apply_label_balance\()",
        "\n",
        "local softmax helper",
    )

    for obsolete in ("def _robust_cost(", "def _stable_softmax("):
        if obsolete in source:
            raise RuntimeError(f"obsolete helper remains: {obsolete}")
    if source.count("GaussianMixtureMeasurementFactor(") != 1:
        raise RuntimeError("expected exactly one PyRecEst factor construction")
    TARGET.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
