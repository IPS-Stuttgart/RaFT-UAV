"""Track-conditioned high-resolution proposal generation.

Established trajectories define uncertainty-aware ROIs.  Each ROI is enlarged,
resampled, evaluated by the stride-4 specialist, and mapped back to native image
coordinates.  Proposal IDs are frame-local; identity is left to the downstream
multi-scan association.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from ._full_stack_io import crop_box, group_id, image_index, load_gray, prediction_files, read_rows, write_json, write_rows
from ._records import Detection
from .tiny_p2_detector import decode, load_detector, _torch

@dataclass(frozen=True)
class RoiConfig:
    history: int = 5
    sigma_scale: float = 3.0
    min_margin: float = 20.0
    box_scale: float = 3.0
    upscale: float = 3.0
    max_roi_side: int = 512
    max_track_gap: int = 18

def predict_roi(history: list[Detection], frame_id: int, image_shape: tuple[int, int], config: RoiConfig) -> tuple[float, float, float, float]:
    past = [r for r in history if r.frame_id < frame_id][-config.history:]
    if not past:
        raise ValueError('ROI prediction requires past observations')
    if frame_id - past[-1].frame_id > config.max_track_gap:
        raise ValueError('track is too stale for conditioned detection')
    times = np.asarray([r.frame_id for r in past], dtype=float)
    centres = np.asarray([[r.center_x, r.center_y] for r in past])
    sizes = np.asarray([[r.width, r.height] for r in past])
    if len(past) >= 2:
        a = np.column_stack((times, np.ones_like(times)))
        coef = np.linalg.lstsq(a, centres, rcond=None)[0]
        centre = np.asarray([frame_id, 1.0]) @ coef
        residual = centres - a @ coef
        sigma = np.sqrt(np.mean(residual ** 2, axis=0) + 1.0) * math.sqrt(max(1, frame_id - past[-1].frame_id))
    else:
        centre = centres[-1]
        sigma = np.asarray([past[-1].width, past[-1].height]) * 0.25 + 1
    size = np.median(sizes, axis=0) * config.box_scale + 2 * config.sigma_scale * sigma + 2 * config.min_margin
    h, w = image_shape
    rw = min(float(config.max_roi_side / config.upscale), max(size[0], 2 * config.min_margin))
    rh = min(float(config.max_roi_side / config.upscale), max(size[1], 2 * config.min_margin))
    x1 = float(np.clip(centre[0] - rw / 2, 0, max(w - rw, 0)))
    y1 = float(np.clip(centre[1] - rh / 2, 0, max(h - rh, 0)))
    return (x1, y1, min(rw, w - x1), min(rh, h - y1))

def generate_track_conditioned(checkpoint: Path, image_root: Path, tracks_dir: Path, output_dir: Path, sequence_names: list[str] | None=None, config: RoiConfig=RoiConfig(), device: str='cpu', score_threshold: float=0.005, top_k: int=40) -> dict:
    torch, _ = _torch()
    model = load_detector(checkpoint, device)
    allowed = None if sequence_names is None else set(sequence_names)
    summary = {}
    for track_file in prediction_files(tracks_dir):
        name = track_file.stem
        if allowed is not None and name not in allowed:
            continue
        frames = image_index(image_root / name)
        tracks = group_id(read_rows(track_file))
        by_frame = {}
        for frame, path in sorted(frames.items()):
            image = load_gray(path)
            frame_rows = []
            for identity, history in tracks.items():
                if not any((r.frame_id < frame for r in history)):
                    continue
                try:
                    x, y, w, h = predict_roi(history, frame, image.shape, config)
                except ValueError:
                    continue
                crop, bounds = crop_box(image, x, y, x + w, y + h)
                target_w = max(16, int(round(crop.shape[1] * config.upscale)))
                target_h = max(16, int(round(crop.shape[0] * config.upscale)))
                resized = np.asarray(Image.fromarray(np.uint8(crop * 255)).resize((target_w, target_h), Image.Resampling.BICUBIC), dtype=np.float32) / 255
                ph = -target_h % 4
                pw = -target_w % 4
                padded = np.pad(resized, ((0, ph), (0, pw)), mode='reflect')
                tensor = torch.from_numpy(padded[None, None]).to(device)
                with torch.no_grad():
                    local = decode(model(tensor), (target_h, target_w), score_threshold, top_k)
                xa, ya, _, _ = bounds
                for lx, ly, lw, lh, score in local:
                    frame_rows.append((xa + lx / config.upscale, ya + ly / config.upscale, lw / config.upscale, lh / config.upscale, score, identity))
            seen = set()
            kept = []
            for row in sorted(frame_rows, key=lambda r: -r[4]):
                key = tuple((round(v, 1) for v in row[:4]))
                if key in seen:
                    continue
                seen.add(key)
                kept.append(row)
            for pid, (x, y, w, h, score, _) in enumerate(kept, 1):
                by_frame.setdefault(frame, []).append(Detection(frame, pid, x, y, w, h, score, 1, 1.0))
        rows = [r for frame in sorted(by_frame) for r in by_frame[frame]]
        write_rows(output_dir / f'{name}.txt', rows)
        summary[name] = len(rows)
    return {'sequences': summary, 'total_proposals': sum(summary.values()), 'config': config.__dict__}

def _names(path: Path | None):
    return None if path is None else [x.strip() for x in path.read_text().splitlines() if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--tracks-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sequence-list', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--score-threshold', type=float, default=0.005)
    p.add_argument('--upscale', type=float, default=3.0)
    p.add_argument('--box-scale', type=float, default=3.0)
    p.add_argument('--summary-json', type=Path)
    return p

def main(argv=None):
    a = build_parser().parse_args(argv)
    result = generate_track_conditioned(a.checkpoint, a.image_root, a.tracks_dir, a.output_dir, _names(a.sequence_list), RoiConfig(upscale=a.upscale, box_scale=a.box_scale), a.device, a.score_threshold)
    if a.summary_json:
        write_json(a.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
