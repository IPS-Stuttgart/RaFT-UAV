"""Stride-4 thermal UAV proposal specialist with size-adaptive NWD supervision.

This detector is deliberately complementary to the full-frame YOLO source.  It
uses a compact stride-4 centre/offset/size head, retains low-confidence tiny-UAV
hypotheses, and exports ordinary nine-column LTS proposal banks.
"""
from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image
from ._full_stack_io import image_index, read_rows, write_json, write_rows
from ._records import Detection

def nwd_distance_np(pred: np.ndarray, target: np.ndarray, eps: float=1e-09) -> np.ndarray:
    """Squared Gaussian-Wasserstein distance for ``[cx,cy,w,h]`` boxes."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    return np.sum((p[..., :2] - t[..., :2]) ** 2, axis=-1) + 0.25 * np.sum((p[..., 2:] - t[..., 2:]) ** 2, axis=-1) + eps

def size_adaptive_nwd_loss_np(pred: np.ndarray, target: np.ndarray, scale: float=12.0, tiny_side: float=24.0) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    side = np.sqrt(np.maximum(target[..., 2] * target[..., 3], 1e-09))
    weight = 1.0 + np.clip((tiny_side - side) / tiny_side, 0.0, 1.0) * 2.0
    return weight * (1.0 - np.exp(-np.sqrt(nwd_distance_np(pred, target)) / scale))

def _torch():
    import torch
    from torch import nn
    return (torch, nn)

class TinyP2Detector:

    def __new__(cls, *args, **kwargs):
        torch, nn = _torch()

        class Residual(nn.Module):

            def __init__(self, c: int):
                super().__init__()
                self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))

            def forward(self, x):
                return torch.nn.functional.silu(x + self.net(x))

        class Model(nn.Module):
            stride = 4

            def __init__(self, channels: int=64, blocks: int=4):
                super().__init__()
                self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU(), nn.Conv2d(32, channels, 3, 2, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(), *[Residual(channels) for _ in range(blocks)])
                self.heat = nn.Conv2d(channels, 1, 1)
                self.offset = nn.Conv2d(channels, 2, 1)
                self.log_size = nn.Conv2d(channels, 2, 1)
                nn.init.constant_(self.heat.bias, -4.0)

            def forward(self, x):
                f = self.stem(x)
                return (self.heat(f), self.offset(f), self.log_size(f))
        return Model(*args, **kwargs)

def _gaussian_target(height: int, width: int, centres: list[tuple[float, float]], sigma: float=1.0) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    for x, y in centres:
        target = np.maximum(target, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)))
    return target

def _tensor_loss(outputs, truth_boxes: list[np.ndarray], stride: int=4, nwd_scale: float=12.0):
    torch, _ = _torch()
    heat, offset, log_size = outputs
    _, _, h, w = heat.shape
    heat_targets = []
    positive_indices = []
    for b, boxes in enumerate(truth_boxes):
        centres = [(float(box[0] / stride), float(box[1] / stride)) for box in boxes]
        heat_targets.append(_gaussian_target(h, w, centres))
        for box in boxes:
            gx = float(box[0] / stride)
            gy = float(box[1] / stride)
            ix = min(w - 1, max(0, int(gx)))
            iy = min(h - 1, max(0, int(gy)))
            positive_indices.append((b, iy, ix, gx - ix, gy - iy, float(box[2]), float(box[3])))
    target = torch.as_tensor(np.stack(heat_targets)[:, None], device=heat.device, dtype=heat.dtype)
    prob = torch.sigmoid(heat).clamp(1e-6, 1 - 1e-6)
    positive_mask = target.eq(1).to(heat.dtype)
    negative_mask = target.lt(1).to(heat.dtype)
    negative_weight = (1 - target).pow(4)
    positive_loss = -torch.log(prob) * (1 - prob).pow(2) * positive_mask
    negative_loss = -torch.log(1 - prob) * prob.pow(2) * negative_weight * negative_mask
    normalizer = positive_mask.sum().clamp(min=1)
    heat_loss = (positive_loss.sum() + negative_loss.sum()) / normalizer
    if not positive_indices:
        return (heat_loss, {'heat': float(heat_loss.detach()), 'offset': 0.0, 'nwd': 0.0})
    off_losses = []
    nwd_losses = []
    for b, iy, ix, ox, oy, tw, th in positive_indices:
        po = offset[b, :, iy, ix]
        ps = torch.exp(torch.clamp(log_size[b, :, iy, ix], -2, 8))
        off_losses.append(torch.nn.functional.smooth_l1_loss(po, torch.tensor([ox, oy], device=po.device), reduction='mean'))
        pred = torch.stack(((ix + po[0]) * stride, (iy + po[1]) * stride, ps[0], ps[1]))
        tgt = torch.tensor([(ix + ox) * stride, (iy + oy) * stride, tw, th], device=pred.device, dtype=pred.dtype)
        dist = (pred[0] - tgt[0]).pow(2) + (pred[1] - tgt[1]).pow(2) + 0.25 * ((pred[2] - tgt[2]).pow(2) + (pred[3] - tgt[3]).pow(2))
        side = torch.sqrt(torch.clamp(tgt[2] * tgt[3], min=1e-06))
        weight = 1 + 2 * torch.clamp((24 - side) / 24, 0, 1)
        nwd_losses.append(weight * (1 - torch.exp(-torch.sqrt(dist + 1e-09) / nwd_scale)))
    off_loss = torch.stack(off_losses).mean()
    nwd_loss = torch.stack(nwd_losses).mean()
    total = heat_loss + off_loss + 2 * nwd_loss
    return (total, {'heat': float(heat_loss.detach()), 'offset': float(off_loss.detach()), 'nwd': float(nwd_loss.detach())})

@dataclass(frozen=True)
class Sample:
    image: Path
    boxes: tuple[tuple[float, float, float, float], ...]

def samples_from_lts(image_root: Path, label_root: Path, sequence_names: Iterable[str] | None=None) -> list[Sample]:
    allowed = None if sequence_names is None else set(sequence_names)
    samples = []
    for label_file in sorted(label_root.glob('*.txt')):
        name = label_file.stem
        if allowed is not None and name not in allowed:
            continue
        seq = image_root / name
        if not seq.is_dir():
            continue
        frames = image_index(seq)
        by_frame = {}
        for row in read_rows(label_file):
            by_frame.setdefault(row.frame_id, []).append((row.center_x, row.center_y, row.width, row.height))
        for frame, path in frames.items():
            samples.append(Sample(path, tuple(by_frame.get(frame, ()))))
    return samples

def _load_tensor(path: Path):
    torch, _ = _torch()
    with Image.open(path) as im:
        arr = np.asarray(im.convert('L'), dtype=np.float32) / 255.0
    h, w = arr.shape
    ph = -h % 4
    pw = -w % 4
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw)), mode='reflect')
    return (torch.from_numpy(arr[None, None]), (h, w))

def train_detector(image_root: Path, label_root: Path, output: Path, sequence_names: list[str] | None=None, epochs: int=20, learning_rate: float=0.002, seed: int=0, device: str='cpu') -> dict:
    torch, _ = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    model = TinyP2Detector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    samples = samples_from_lts(image_root, label_root, sequence_names)
    if not samples:
        raise ValueError('no P2 training samples')
    history = []
    for _ in range(epochs):
        random.shuffle(samples)
        totals = []
        model.train()
        for sample in samples:
            tensor, _ = _load_tensor(sample.image)
            tensor = tensor.to(device)
            boxes = [np.asarray(sample.boxes, dtype=np.float32).reshape(-1, 4)]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _tensor_loss(model(tensor), boxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            totals.append(float(loss.detach()))
        history.append(float(np.mean(totals)))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'stride': 4, 'history': history, 'format': 'raft-uav-tiny-p2-v1'}, output)
    return {'sample_count': len(samples), 'epochs': epochs, 'final_loss': history[-1], 'checkpoint': str(output)}

def load_detector(checkpoint: Path, device: str='cpu'):
    torch, _ = _torch()
    payload = torch.load(checkpoint, map_location=device)
    model = TinyP2Detector().to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model

def decode(outputs, native_shape: tuple[int, int], score_threshold: float=0.01, top_k: int=300, stride: int=4) -> list[tuple[float, float, float, float, float]]:
    torch, _ = _torch()
    heat, offset, log_size = outputs
    scores = torch.sigmoid(heat[0, 0])
    pooled = torch.nn.functional.max_pool2d(scores[None, None], 3, 1, 1)[0, 0]
    scores = torch.where(scores >= pooled, scores, torch.zeros_like(scores))
    flat = scores.flatten()
    k = min(top_k, flat.numel())
    values, indices = torch.topk(flat, k)
    h, w = native_shape
    out = []
    for score, index in zip(values.tolist(), indices.tolist()):
        if score < score_threshold:
            break
        iy = index // scores.shape[1]
        ix = index % scores.shape[1]
        off = offset[0, :, iy, ix]
        size = torch.exp(torch.clamp(log_size[0, :, iy, ix], -2, 8))
        cx = (ix + float(off[0])) * stride
        cy = (iy + float(off[1])) * stride
        bw = float(size[0])
        bh = float(size[1])
        x1 = max(0, cx - bw / 2)
        y1 = max(0, cy - bh / 2)
        bw = min(bw, w - x1)
        bh = min(bh, h - y1)
        if bw >= 1 and bh >= 1:
            out.append((x1, y1, bw, bh, score))
    return out

def predict_sequences(checkpoint: Path, image_root: Path, output_dir: Path, sequence_names: list[str] | None=None, device: str='cpu', score_threshold: float=0.01, top_k: int=300) -> dict:
    torch, _ = _torch()
    model = load_detector(checkpoint, device)
    allowed = None if sequence_names is None else set(sequence_names)
    summary = {}
    for seq in sorted((p for p in image_root.iterdir() if p.is_dir())):
        if allowed is not None and seq.name not in allowed:
            continue
        rows = []
        for frame, path in sorted(image_index(seq).items()):
            tensor, shape = _load_tensor(path)
            with torch.no_grad():
                boxes = decode(model(tensor.to(device)), shape, score_threshold, top_k)
            for local_id, (x, y, w, h, score) in enumerate(boxes, 1):
                rows.append(Detection(frame, local_id, x, y, w, h, score, 1, 1.0))
        write_rows(output_dir / f'{seq.name}.txt', rows)
        summary[seq.name] = len(rows)
    return {'sequences': summary, 'total_proposals': sum(summary.values())}

def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--image-root', type=Path, required=True)
    tr.add_argument('--label-root', type=Path, required=True)
    tr.add_argument('--output', type=Path, required=True)
    tr.add_argument('--sequence-list', type=Path)
    tr.add_argument('--epochs', type=int, default=20)
    tr.add_argument('--learning-rate', type=float, default=0.002)
    tr.add_argument('--device', default='cpu')
    tr.add_argument('--summary-json', type=Path)
    pr = sub.add_parser('predict')
    pr.add_argument('--checkpoint', type=Path, required=True)
    pr.add_argument('--image-root', type=Path, required=True)
    pr.add_argument('--output-dir', type=Path, required=True)
    pr.add_argument('--sequence-list', type=Path)
    pr.add_argument('--device', default='cpu')
    pr.add_argument('--score-threshold', type=float, default=0.01)
    pr.add_argument('--top-k', type=int, default=300)
    pr.add_argument('--summary-json', type=Path)
    return p

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == 'train':
        result = train_detector(args.image_root, args.label_root, args.output, _names(args.sequence_list), args.epochs, args.learning_rate, device=args.device)
    else:
        result = predict_sequences(args.checkpoint, args.image_root, args.output_dir, _names(args.sequence_list), args.device, args.score_threshold, args.top_k)
    if args.summary_json:
        write_json(args.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
