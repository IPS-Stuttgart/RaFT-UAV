"""Leakage-safe sequence-level mixture of tracking experts.

The gate uses only observable early-sequence image, seed, and raw-tracker
statistics.  Candidate scores are consumed only while fitting on complementary
folds; application copies one complete prediction file per sequence.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ._full_stack_io import image_index, load_gray, read_rows, write_json
from .temporal_roi_proposals import phase_translation
FEATURE_NAMES = ('log_seed_count', 'median_seed_side', 'seed_density', 'image_contrast', 'image_entropy', 'camera_motion', 'raw_density', 'raw_confidence', 'raw_box_side', 'raw_fragmentation')

@dataclass(frozen=True)
class ExpertGateModel:
    candidates: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    residual_scale: np.ndarray
    margin: float = 0.0005

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = (np.asarray(features) - self.mean) / self.scale
        return z @ self.coefficients.T + self.intercept

    def choose(self, features: np.ndarray) -> str:
        gain = self.predict(features)
        index = int(np.argmax(gain))
        return self.candidates[index] if gain[index] - 1.645 * self.residual_scale[index] > self.margin else 'raw'

    def save(self, path: Path):
        write_json(path, {'format': 'raft-uav-observable-gate-v1', 'candidates': list(self.candidates), 'mean': self.mean.tolist(), 'scale': self.scale.tolist(), 'coefficients': self.coefficients.tolist(), 'intercept': self.intercept.tolist(), 'residual_scale': self.residual_scale.tolist(), 'margin': self.margin, 'feature_names': list(FEATURE_NAMES)})

    @classmethod
    def load(cls, path: Path):
        p = json.loads(path.read_text())
        return cls(tuple(p['candidates']), np.asarray(p['mean']), np.asarray(p['scale']), np.asarray(p['coefficients']), np.asarray(p['intercept']), np.asarray(p['residual_scale']), float(p['margin']))

def _entropy(image: np.ndarray) -> float:
    hist = np.histogram(image, bins=32, range=(0, 1), density=True)[0]
    prob = hist / max(hist.sum(), 1e-09)
    return float(-np.sum(prob * np.log(prob + 1e-12)))

def sequence_features(name: str, image_root: Path, seed_dir: Path, raw_dir: Path, early_frames: int=30) -> np.ndarray:
    seed = read_rows(seed_dir / f'{name}.txt')
    raw = read_rows(raw_dir / f'{name}.txt')
    paths = image_index(image_root / name)
    selected = sorted(paths)[:early_frames]
    images = [load_gray(paths[f]) for f in selected]
    contrast = float(np.median([np.std(i) for i in images])) if images else 0
    entropy = float(np.median([_entropy(i) for i in images])) if images else 0
    motion = []
    for a, b in zip(images, images[1:]):
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        dy, dx = phase_translation(a[:h, :w], b[:h, :w])
        motion.append(math.hypot(dx, dy) / max(math.hypot(w, h), 1))
    early = [r for r in raw if r.frame_id <= early_frames]
    ids = {r.object_id for r in early}
    observed_frames = max(len(selected), 1)
    seed_sides = [math.sqrt(r.width * r.height) for r in seed]
    raw_sides = [math.sqrt(r.width * r.height) for r in early]
    return np.asarray([math.log1p(len(seed)), float(np.median(seed_sides) if seed_sides else 0), len(seed) / max(images[0].size if images else 1, 1), contrast, entropy, float(np.median(motion) if motion else 0), len(early) / observed_frames, float(np.mean([r.confidence for r in early]) if early else 0), float(np.median(raw_sides) if raw_sides else 0), len(ids) / max(len(seed), 1)], float)

def fit_gate(features: dict[str, np.ndarray], scores: dict[str, dict[str, float]], raw_name: str='raw', l2: float=3.0, margin: float=0.0005) -> ExpertGateModel:
    sequences = sorted(set(features) & set(scores))
    candidates = sorted({c for s in sequences for c in scores[s] if c != raw_name})
    x = np.stack([features[s] for s in sequences])
    mean = x.mean(0)
    scale = x.std(0)
    scale = np.where(scale < 1e-06, 1, scale)
    z = (x - mean) / scale
    design = np.column_stack((z, np.ones(len(z))))
    penalty = np.eye(design.shape[1]) * l2
    penalty[-1, -1] = 0
    coef = []
    intercept = []
    residual = []
    for candidate in candidates:
        y = np.asarray([scores[s].get(candidate, scores[s][raw_name]) - scores[s][raw_name] for s in sequences])
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        prediction = design @ weights
        coef.append(weights[:-1])
        intercept.append(weights[-1])
        residual.append(float(np.sqrt(np.mean((y - prediction) ** 2) + 1e-10)))
    return ExpertGateModel(tuple(candidates), mean, scale, np.asarray(coef), np.asarray(intercept), np.asarray(residual), margin)

def read_score_csv(path: Path) -> dict[str, dict[str, float]]:
    out = {}
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            out.setdefault(row['sequence'], {})[row['candidate']] = float(row['score'])
    return out

def apply_gate(model: ExpertGateModel, image_root: Path, seed_dir: Path, raw_dir: Path, candidate_dirs: dict[str, Path], output_dir: Path, sequence_names: list[str] | None=None) -> dict:
    names = sequence_names or sorted((p.stem for p in raw_dir.glob('*.txt')))
    output_dir.mkdir(parents=True, exist_ok=True)
    choices = {}
    for name in names:
        feature = sequence_features(name, image_root, seed_dir, raw_dir)
        choice = model.choose(feature)
        source = raw_dir if choice == 'raw' else candidate_dirs.get(choice, raw_dir)
        file = source / f'{name}.txt'
        if not file.is_file():
            choice = 'raw'
            file = raw_dir / f'{name}.txt'
        shutil.copy2(file, output_dir / file.name)
        choices[name] = {'expert': choice, 'features': feature.tolist(), 'predicted_gains': model.predict(feature).tolist()}
    return {'choices': choices, 'counts': {c: sum((v['expert'] == c for v in choices.values())) for c in ['raw', *model.candidates]}}

def _parse_candidate(values: list[str]) -> dict[str, Path]:
    out = {}
    for value in values:
        name, path = value.split('=', 1)
        out[name] = Path(path)
    return out

def _names(path):
    return None if path is None else [x.strip() for x in path.read_text().splitlines() if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--score-csv', type=Path, required=True)
    fit.add_argument('--image-root', type=Path, required=True)
    fit.add_argument('--seed-dir', type=Path, required=True)
    fit.add_argument('--raw-dir', type=Path, required=True)
    fit.add_argument('--output-model', type=Path, required=True)
    fit.add_argument('--margin', type=float, default=0.0005)
    app = sub.add_parser('apply')
    app.add_argument('--model', type=Path, required=True)
    app.add_argument('--image-root', type=Path, required=True)
    app.add_argument('--seed-dir', type=Path, required=True)
    app.add_argument('--raw-dir', type=Path, required=True)
    app.add_argument('--candidate', action='append', default=[])
    app.add_argument('--output-dir', type=Path, required=True)
    app.add_argument('--sequence-list', type=Path)
    app.add_argument('--summary-json', type=Path)
    return p

def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.command == 'fit':
        scores = read_score_csv(a.score_csv)
        features = {s: sequence_features(s, a.image_root, a.seed_dir, a.raw_dir) for s in scores}
        model = fit_gate(features, scores, margin=a.margin)
        model.save(a.output_model)
        print(json.dumps({'model': str(a.output_model), 'sequences': len(features), 'candidates': list(model.candidates)}, indent=2))
        return 0
    model = ExpertGateModel.load(a.model)
    result = apply_gate(model, a.image_root, a.seed_dir, a.raw_dir, _parse_candidate(a.candidate), a.output_dir, _names(a.sequence_list))
    if a.summary_json:
        write_json(a.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
