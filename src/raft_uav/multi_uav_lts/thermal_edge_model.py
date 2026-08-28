"""Task-specific tiny-thermal affinity model with hard-negative training."""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn
from ._full_stack_io import crop_box, group_frame, group_id, image_index, load_gray, read_rows, write_json
from ._records import Detection

@dataclass(frozen=True)
class ThermalModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    feature_names: tuple[str, ...]

    def probability(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features) - self.mean) / self.scale
        logits = np.clip(x @ self.weights[:-1] + self.weights[-1], -40, 40)
        return 1 / (1 + np.exp(-logits))

    def save(self, path: Path) -> None:
        write_json(path, {'format': 'raft-uav-thermal-affinity-v1', 'mean': self.mean.tolist(), 'scale': self.scale.tolist(), 'weights': self.weights.tolist(), 'feature_names': list(self.feature_names)})

    @classmethod
    def load(cls, path: Path) -> 'ThermalModel':
        p = json.loads(path.read_text())
        return cls(np.asarray(p['mean']), np.asarray(p['scale']), np.asarray(p['weights']), tuple(p['feature_names']))
FEATURE_NAMES = ('ncc', 'mad', 'grad_ncc', 'hist_l1', 'dct_l1', 'ring_delta', 'log_area_ratio', 'aspect_delta', 'normalized_motion', 'gap')

def _resize(array: np.ndarray, size: int=32) -> np.ndarray:
    if array.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    return np.asarray(Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255

def _znorm(x: np.ndarray) -> np.ndarray:
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-06)

def _crop(image: np.ndarray, row: Detection, scale: float) -> np.ndarray:
    cx = row.center_x
    cy = row.center_y
    w = row.width * scale
    h = row.height * scale
    return crop_box(image, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)[0]

def pair_features(image_a: np.ndarray, a: Detection, image_b: np.ndarray, b: Detection) -> np.ndarray:
    pa = _znorm(_resize(_crop(image_a, a, 1.5)))
    pb = _znorm(_resize(_crop(image_b, b, 1.5)))
    ncc = float(np.mean(pa * pb))
    mad = float(np.mean(np.abs(pa - pb)))
    ga = np.hypot(*np.gradient(pa))
    gb = np.hypot(*np.gradient(pb))
    grad_ncc = float(np.mean(_znorm(ga) * _znorm(gb)))
    ha = np.histogram(pa, bins=16, range=(-3, 3), density=True)[0]
    hb = np.histogram(pb, bins=16, range=(-3, 3), density=True)[0]
    hist_l1 = float(np.mean(np.abs(ha - hb)))
    da = dctn(pa, norm='ortho')[:6, :6]
    db = dctn(pb, norm='ortho')[:6, :6]
    dct_l1 = float(np.mean(np.abs(da - db)))
    ca = _resize(_crop(image_a, a, 1.0))
    ra = _resize(_crop(image_a, a, 2.5))
    cb = _resize(_crop(image_b, b, 1.0))
    rb = _resize(_crop(image_b, b, 2.5))
    ring_delta = abs(float(np.mean(ra) - np.mean(ca)) - float(np.mean(rb) - np.mean(cb)))
    log_area = abs(math.log(max(b.width * b.height, 1e-06) / max(a.width * a.height, 1e-06)))
    aspect = abs(math.log(max(b.width / b.height, 1e-06) / max(a.width / a.height, 1e-06)))
    gap = max(1, b.frame_id - a.frame_id)
    motion = math.hypot(b.center_x - a.center_x, b.center_y - a.center_y) / (max(math.sqrt(a.width * a.height), 2) * gap)
    return np.asarray([ncc, mad, grad_ncc, hist_l1, dct_l1, ring_delta, log_area, aspect, motion, float(gap)], dtype=float)

def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float=1.0, iterations: int=80) -> ThermalModel:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mean = x.mean(0)
    scale = x.std(0)
    scale = np.where(scale < 1e-06, 1, scale)
    z = (x - mean) / scale
    design = np.column_stack((z, np.ones(len(z))))
    weights = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[-1, -1] = 0
    for _ in range(iterations):
        logits = np.clip(design @ weights, -40, 40)
        p = 1 / (1 + np.exp(-logits))
        curvature = np.maximum(p * (1 - p), 1e-05)
        grad = design.T @ (p - y) + penalty @ weights
        hessian = design.T * curvature @ design + penalty
        step = np.linalg.solve(hessian, grad)
        weights -= step
        if np.linalg.norm(step) < 1e-07:
            break
    return ThermalModel(mean, scale, weights, FEATURE_NAMES)

def build_training_set(image_root: Path, label_root: Path, sequence_names: list[str] | None=None, max_gap: int=8, max_pairs: int=100000, seed: int=0) -> tuple[np.ndarray, np.ndarray, dict]:
    allowed = None if sequence_names is None else set(sequence_names)
    rng = np.random.default_rng(seed)
    features = []
    labels = []
    positives = negatives = 0
    for label_file in sorted(label_root.glob('*.txt')):
        name = label_file.stem
        if allowed is not None and name not in allowed:
            continue
        seq = image_root / name
        if not seq.is_dir():
            continue
        paths = image_index(seq)
        images = {}
        rows = read_rows(label_file)
        tracks = group_id(rows)
        by_frame = group_frame(rows)

        def image(frame):
            if frame not in images:
                images[frame] = load_gray(paths[frame])
            return images[frame]
        for identity, track in tracks.items():
            for i, a in enumerate(track):
                for b in track[i + 1:]:
                    if b.frame_id - a.frame_id > max_gap:
                        break
                    if a.frame_id not in paths or b.frame_id not in paths:
                        continue
                    features.append(pair_features(image(a.frame_id), a, image(b.frame_id), b))
                    labels.append(1)
                    positives += 1
                    candidates = [r for r in by_frame.get(b.frame_id, []) if r.object_id != identity]
                    if candidates:
                        hard = min(candidates, key=lambda r: math.hypot(r.center_x - b.center_x, r.center_y - b.center_y))
                        features.append(pair_features(image(a.frame_id), a, image(hard.frame_id), hard))
                        labels.append(0)
                        negatives += 1
    if not features:
        raise ValueError('no thermal training pairs')
    x = np.asarray(features)
    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        raise ValueError('thermal training requires both positive and hard-negative pairs')
    if len(y) > max_pairs:
        indices = rng.choice(len(y), max_pairs, replace=False)
        x = x[indices]
        y = y[indices]
    return (x, y, {'positive_pairs': positives, 'negative_pairs': negatives, 'retained_pairs': len(y)})

def train_model(image_root: Path, label_root: Path, output: Path, sequence_names: list[str] | None=None, max_gap: int=8, l2: float=1.0) -> dict:
    x, y, summary = build_training_set(image_root, label_root, sequence_names, max_gap)
    model = fit_logistic(x, y, l2)
    model.save(output)
    probabilities = model.probability(x)
    summary.update({'model': str(output), 'training_accuracy': float(np.mean((probabilities >= 0.5) == y))})
    return summary

def make_affinity(model: ThermalModel, image_paths: dict[int, Path]):
    cache = {}

    def image(frame):
        if frame not in cache:
            cache[frame] = load_gray(image_paths[frame])
        return cache[frame]

    def affinity(a: Detection, b: Detection) -> float:
        if a.frame_id not in image_paths or b.frame_id not in image_paths:
            return 0.0
        p = float(model.probability(pair_features(image(a.frame_id), a, image(b.frame_id), b)))
        return math.log(max(p, 1e-06)) - math.log(max(1 - p, 1e-06))
    return affinity

def _names(path):
    return None if path is None else [x.strip() for x in path.read_text().splitlines() if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--label-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence-list', type=Path)
    p.add_argument('--max-gap', type=int, default=8)
    p.add_argument('--l2', type=float, default=1)
    p.add_argument('--summary-json', type=Path)
    return p

def main(argv=None):
    a = build_parser().parse_args(argv)
    result = train_model(a.image_root, a.label_root, a.output, _names(a.sequence_list), a.max_gap, a.l2)
    if a.summary_json:
        write_json(a.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
