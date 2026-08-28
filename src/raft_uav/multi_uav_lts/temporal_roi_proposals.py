"""Motion-compensated multi-frame residual proposals inside predicted target ROIs."""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import ndimage
from ._full_stack_io import crop_box, group_id, image_index, load_gray, prediction_files, read_rows, write_json, write_rows
from ._records import Detection
from .track_conditioned_proposals import RoiConfig, predict_roi

@dataclass(frozen=True)
class TemporalRoiConfig:
    radius: int = 2
    robust_z: float = 3.5
    min_area: int = 2
    max_area: int = 900
    centre_gate: float = 0.65
    roi: RoiConfig = RoiConfig(box_scale=4.0, upscale=1.0)

def phase_translation(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    h = min(reference.shape[0], moving.shape[0])
    w = min(reference.shape[1], moving.shape[1])
    a = reference[:h, :w] - np.mean(reference[:h, :w])
    b = moving[:h, :w] - np.mean(moving[:h, :w])
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    cross /= np.maximum(np.abs(cross), 1e-09)
    corr = np.fft.ifft2(cross).real
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    if y > h // 2:
        y -= h
    if x > w // 2:
        x -= w
    return (int(y), int(x))

def residual_components(current: np.ndarray, neighbours: list[np.ndarray], config: TemporalRoiConfig) -> list[tuple[int, int, int, int, float]]:
    aligned = []
    height, width = current.shape
    y0, y1 = int(0.3 * height), int(0.7 * height)
    x0, x1 = int(0.3 * width), int(0.7 * width)
    background_reference = current.copy()
    background_reference[y0:y1, x0:x1] = 0
    for image in neighbours:
        background_image = image.copy()
        background_image[y0:y1, x0:x1] = 0
        if np.std(background_reference) < 1e-5 or np.std(background_image) < 1e-5:
            dy, dx = (0, 0)
        else:
            dy, dx = phase_translation(background_reference, background_image)
        aligned.append(np.roll(image, (dy, dx), (0, 1)))
    if not aligned:
        return []
    background = np.median(np.stack(aligned), axis=0)
    residual = np.abs(current - background)
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med))) + 1e-06
    threshold = med + config.robust_z * 1.4826 * mad
    mask = residual > threshold
    labels, _ = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    out = []
    for label_id, slices in enumerate(objects, 1):
        if slices is None:
            continue
        ys, xs = slices
        area = int(np.sum(labels[slices] == label_id))
        if not config.min_area <= area <= config.max_area:
            continue
        score = float(np.mean(residual[slices][labels[slices] == label_id]) / (threshold + 1e-06))
        out.append((xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start, min(score, 1.0)))
    return out

def generate_temporal_roi(image_root: Path, tracks_dir: Path, output_dir: Path, sequence_names: list[str] | None=None, config: TemporalRoiConfig=TemporalRoiConfig()) -> dict:
    allowed = None if sequence_names is None else set(sequence_names)
    summary = {}
    for track_file in prediction_files(tracks_dir):
        name = track_file.stem
        if allowed is not None and name not in allowed:
            continue
        frames = image_index(image_root / name)
        cache = {frame: load_gray(path) for frame, path in frames.items()}
        tracks = group_id(read_rows(track_file))
        proposals = []
        for frame, image in sorted(cache.items()):
            local = []
            for _, history in tracks.items():
                if not any((r.frame_id < frame for r in history)):
                    continue
                try:
                    x, y, w, h = predict_roi(history, frame, image.shape, config.roi)
                except ValueError:
                    continue
                current, bounds = crop_box(image, x, y, x + w, y + h)
                neighbours = []
                for other in range(max(1, frame - config.radius), frame + config.radius + 1):
                    if other == frame or other not in cache:
                        continue
                    crop, _ = crop_box(cache[other], x, y, x + w, y + h)
                    hh = min(current.shape[0], crop.shape[0])
                    ww = min(current.shape[1], crop.shape[1])
                    neighbours.append(crop[:hh, :ww])
                if not neighbours:
                    continue
                hh = min([current.shape[0], *[n.shape[0] for n in neighbours]])
                ww = min([current.shape[1], *[n.shape[1] for n in neighbours]])
                current = current[:hh, :ww]
                neighbours = [n[:hh, :ww] for n in neighbours]
                cx = w / 2
                cy = h / 2
                for bx, by, bw, bh, score in residual_components(current, neighbours, config):
                    dc = np.hypot((bx + bw / 2 - cx) / max(w, 1), (by + bh / 2 - cy) / max(h, 1))
                    if dc <= config.centre_gate:
                        local.append((bounds[0] + bx, bounds[1] + by, bw, bh, score))
            for pid, (x, y, w, h, score) in enumerate(sorted(local, key=lambda r: -r[4]), 1):
                proposals.append(Detection(frame, pid, x, y, w, h, score, 1, 1.0))
        write_rows(output_dir / f'{name}.txt', proposals)
        summary[name] = len(proposals)
    return {'sequences': summary, 'total_proposals': sum(summary.values()), 'config': {'radius': config.radius, 'robust_z': config.robust_z}}

def _names(path):
    return None if path is None else [x.strip() for x in path.read_text().splitlines() if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--tracks-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sequence-list', type=Path)
    p.add_argument('--radius', type=int, default=2)
    p.add_argument('--robust-z', type=float, default=3.5)
    p.add_argument('--summary-json', type=Path)
    return p

def main(argv=None):
    a = build_parser().parse_args(argv)
    result = generate_temporal_roi(a.image_root, a.tracks_dir, a.output_dir, _names(a.sequence_list), TemporalRoiConfig(radius=a.radius, robust_z=a.robust_z))
    if a.summary_json:
        write_json(a.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
