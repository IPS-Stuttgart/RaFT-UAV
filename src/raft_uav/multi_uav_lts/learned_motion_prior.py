"""Leakage-safe learned transition density for tiny-UAV proposal association."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.mixture import GaussianMixture

from ._full_stack_io import group_id, read_rows, write_json
from ._records import Detection


FEATURE_NAMES = (
    "dx_scaled",
    "dy_scaled",
    "log_width_ratio",
    "log_height_ratio",
    "log_area",
    "gap",
)


@dataclass(frozen=True)
class MotionPrior:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    score_center: float
    score_scale: float
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def _standardized(self, a: Detection, b: Detection) -> np.ndarray:
        return (transition_features(a, b) - self.feature_mean) / self.feature_scale

    def log_density(self, a: Detection, b: Detection) -> float:
        point = self._standardized(a, b)
        terms = []
        dimension = len(point)
        for weight, mean, covariance in zip(
            self.weights,
            self.means,
            self.covariances,
            strict=True,
        ):
            covariance = np.asarray(covariance, dtype=float)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                continue
            difference = point - mean
            mahalanobis = float(difference @ np.linalg.pinv(covariance) @ difference)
            terms.append(
                math.log(max(float(weight), 1e-12))
                - 0.5 * (dimension * math.log(2.0 * math.pi) + logdet + mahalanobis)
            )
        if not terms:
            return -100.0
        maximum = max(terms)
        return maximum + math.log(sum(math.exp(value - maximum) for value in terms))

    def similarity(self, a: Detection, b: Detection) -> float:
        log_density = self.log_density(a, b)
        standardized = (log_density - self.score_center) / max(self.score_scale, 1e-6)
        return float(1.0 / (1.0 + math.exp(-float(np.clip(standardized, -30.0, 30.0)))))

    def save(self, path: Path) -> None:
        write_json(
            path,
            {
                "format": "raft-uav-motion-prior-v1",
                "weights": self.weights.tolist(),
                "means": self.means.tolist(),
                "covariances": self.covariances.tolist(),
                "feature_mean": self.feature_mean.tolist(),
                "feature_scale": self.feature_scale.tolist(),
                "score_center": self.score_center,
                "score_scale": self.score_scale,
                "feature_names": list(self.feature_names),
            },
        )

    @classmethod
    def load(cls, path: Path) -> "MotionPrior":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "raft-uav-motion-prior-v1":
            raise ValueError(f"unsupported motion-prior format in {path}")
        return cls(
            np.asarray(payload["weights"], dtype=float),
            np.asarray(payload["means"], dtype=float),
            np.asarray(payload["covariances"], dtype=float),
            np.asarray(payload["feature_mean"], dtype=float),
            np.asarray(payload["feature_scale"], dtype=float),
            float(payload["score_center"]),
            float(payload["score_scale"]),
            tuple(payload["feature_names"]),
        )


def transition_features(a: Detection, b: Detection) -> np.ndarray:
    gap = b.frame_id - a.frame_id
    if gap <= 0:
        raise ValueError("motion-prior transitions must advance in time")
    scale = max(math.sqrt(max(a.width * a.height, 1e-6)), 2.0)
    time_scale = math.sqrt(gap)
    return np.asarray(
        [
            (b.center_x - a.center_x) / (scale * time_scale),
            (b.center_y - a.center_y) / (scale * time_scale),
            math.log(max(b.width, 1e-6) / max(a.width, 1e-6)) / time_scale,
            math.log(max(b.height, 1e-6) / max(a.height, 1e-6)) / time_scale,
            math.log(max(a.width * a.height, 1e-6)),
            math.log1p(gap - 1),
        ],
        dtype=float,
    )


def _parse_translations(
    translations: object,
    *,
    source: Path,
    sequence: str,
) -> dict[int, tuple[float, float]]:
    if not isinstance(translations, dict):
        raise ValueError(f"missing translations for {sequence} in {source}")
    return {
        int(frame): (float(values[0]), float(values[1]))
        for frame, values in translations.items()
    }


def load_translation_maps(
    path: Path | None,
) -> dict[str, dict[int, tuple[float, float]]]:
    if path is None:
        return {}
    if path.is_dir():
        output = {}
        for file in sorted(path.glob("*.json")):
            payload = json.loads(file.read_text(encoding="utf-8"))
            sequence = str(payload.get("sequence") or file.stem)
            output[sequence] = _parse_translations(
                payload.get("translations"),
                source=file,
                sequence=sequence,
            )
        return output
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = payload.get("sequences")
    if not isinstance(sequences, dict):
        raise ValueError(f"stabilization cache has no sequence map: {path}")
    output: dict[str, dict[int, tuple[float, float]]] = {}
    for name, sequence in sequences.items():
        translations = sequence.get("translations") if isinstance(sequence, dict) else None
        output[str(name)] = _parse_translations(
            translations,
            source=path,
            sequence=str(name),
        )
    return output


def load_sequence_translations(
    path: Path,
    sequence: str,
    preloaded: dict[str, dict[int, tuple[float, float]]] | None = None,
) -> dict[int, tuple[float, float]]:
    if path.is_dir():
        file = path / f"{sequence}.json"
        if not file.is_file():
            raise ValueError(f"stabilization cache is missing sequence {sequence}")
        payload = json.loads(file.read_text(encoding="utf-8"))
        return _parse_translations(
            payload.get("translations"),
            source=file,
            sequence=sequence,
        )
    values = preloaded if preloaded is not None else load_translation_maps(path)
    if sequence not in values:
        raise ValueError(f"stabilization cache is missing sequence {sequence}")
    return values[sequence]


def _stabilize_rows(
    rows: list[Detection],
    translations: dict[int, tuple[float, float]],
) -> list[Detection]:
    output = []
    for row in rows:
        dy, dx = translations.get(row.frame_id, (0.0, 0.0))
        output.append(
            Detection(
                row.frame_id,
                row.object_id,
                row.x1 + dx,
                row.y1 + dy,
                row.width,
                row.height,
                row.confidence,
                row.class_id,
                row.visibility,
            )
        )
    return output


def collect_transitions(
    truth_dir: Path,
    sequence_names: list[str] | None = None,
    *,
    max_gap: int = 4,
    max_samples: int = 200_000,
    seed: int = 0,
    stabilization_file: Path | None = None,
) -> tuple[np.ndarray, dict]:
    allowed = None if sequence_names is None else set(sequence_names)
    translation_maps = (
        load_translation_maps(stabilization_file)
        if stabilization_file is not None and stabilization_file.is_file()
        else {}
    )
    features: list[np.ndarray] = []
    sequence_counts: dict[str, int] = {}
    for truth_file in sorted(truth_dir.glob("*.txt")):
        if allowed is not None and truth_file.stem not in allowed:
            continue
        retained = 0
        rows = read_rows(truth_file)
        if stabilization_file is not None:
            translations = load_sequence_translations(
                stabilization_file,
                truth_file.stem,
                translation_maps,
            )
            rows = _stabilize_rows(rows, translations)
        for track in group_id(rows).values():
            for index, left in enumerate(track):
                for right in track[index + 1 :]:
                    gap = right.frame_id - left.frame_id
                    if gap > max_gap:
                        break
                    features.append(transition_features(left, right))
                    retained += 1
        sequence_counts[truth_file.stem] = retained
    if not features:
        raise ValueError("no motion-prior transitions were collected")
    matrix = np.asarray(features, dtype=float)
    if len(matrix) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(matrix), max_samples, replace=False)
        matrix = matrix[indices]
    return matrix, {
        "sequence_counts": sequence_counts,
        "available_samples": len(features),
        "retained_samples": len(matrix),
        "max_gap": max_gap,
        "stabilized": stabilization_file is not None,
        "stabilization_file": None
        if stabilization_file is None
        else str(stabilization_file),
    }


def fit_prior(
    features: np.ndarray,
    *,
    components: int = 4,
    random_state: int = 0,
    reg_covar: float = 1e-3,
) -> MotionPrior:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected a matrix with {len(FEATURE_NAMES)} columns")
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
    standardized = (matrix - feature_mean) / feature_scale
    component_count = max(1, min(int(components), len(standardized)))
    model = GaussianMixture(
        n_components=component_count,
        covariance_type="full",
        reg_covar=reg_covar,
        random_state=random_state,
        n_init=2,
        max_iter=200,
    )
    model.fit(standardized)
    scores = model.score_samples(standardized)
    center = float(np.median(scores))
    scale = float(1.4826 * np.median(np.abs(scores - center)))
    if scale < 1e-6:
        scale = float(np.std(scores) + 1e-6)
    return MotionPrior(
        model.weights_.copy(),
        model.means_.copy(),
        model.covariances_.copy(),
        feature_mean,
        feature_scale,
        center,
        scale,
    )


def train_prior(
    truth_dir: Path,
    output: Path,
    sequence_names: list[str] | None = None,
    *,
    components: int = 4,
    max_gap: int = 4,
    stabilization_file: Path | None = None,
) -> dict:
    features, summary = collect_transitions(
        truth_dir,
        sequence_names,
        max_gap=max_gap,
        stabilization_file=stabilization_file,
    )
    prior = fit_prior(features, components=components)
    prior.save(output)
    similarities = []
    for feature in features[: min(len(features), 20_000)]:
        # Use the fitted model directly for a deterministic training-density summary.
        standardized = (feature - prior.feature_mean) / prior.feature_scale
        log_terms = []
        for weight, mean, covariance in zip(
            prior.weights,
            prior.means,
            prior.covariances,
            strict=True,
        ):
            difference = standardized - mean
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                continue
            log_terms.append(
                math.log(max(float(weight), 1e-12))
                - 0.5
                * (
                    len(feature) * math.log(2.0 * math.pi)
                    + logdet
                    + float(difference @ np.linalg.pinv(covariance) @ difference)
                )
            )
        if log_terms:
            maximum = max(log_terms)
            density = maximum + math.log(sum(math.exp(value - maximum) for value in log_terms))
            similarities.append(
                1.0
                / (
                    1.0
                    + math.exp(
                        -float(
                            np.clip(
                                (density - prior.score_center) / prior.score_scale,
                                -30.0,
                                30.0,
                            )
                        )
                    )
                )
            )
    summary.update(
        {
            "model": str(output),
            "components": len(prior.weights),
            "median_training_similarity": float(np.median(similarities)),
        }
    )
    return summary


def make_affinity(prior: MotionPrior) -> Callable[[Detection, Detection], float]:
    def affinity(a: Detection, b: Detection) -> float:
        probability = float(np.clip(prior.similarity(a, b), 1e-6, 1.0 - 1e-6))
        return math.log(probability) - math.log1p(-probability)

    return affinity


def map_affinity(
    affinity: Callable[[Detection, Detection], float],
    geometry: Callable[[Detection], Detection],
) -> Callable[[Detection, Detection], float]:
    """Evaluate a learned motion affinity in an alternative coordinate system."""

    def mapped(a: Detection, b: Detection) -> float:
        return affinity(geometry(a), geometry(b))

    return mapped


def combine_affinities(
    *weighted: tuple[float, Callable[[Detection, Detection], float]],
) -> Callable[[Detection, Detection], float]:
    total_weight = sum(max(float(weight), 0.0) for weight, _ in weighted)
    if total_weight <= 0:
        raise ValueError("combined affinity requires a positive weight")

    def combined(a: Detection, b: Detection) -> float:
        return sum(max(float(weight), 0.0) * function(a, b) for weight, function in weighted) / total_weight

    return combined


def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-list", type=Path)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--stabilization-file", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_prior(
        args.truth_dir,
        args.output,
        _names(args.sequence_list),
        components=args.components,
        max_gap=args.max_gap,
        stabilization_file=args.stabilization_file,
    )
    if args.summary_json:
        write_json(args.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
