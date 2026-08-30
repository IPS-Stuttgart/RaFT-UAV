"""Symmetric multi-frame P2 proposal specialist for tiny thermal UAVs.

The model is trained only on compact, label-centred patches.  At inference it
runs inside track-conditioned ROIs, so it can use future and past registered
frames without the cost of applying a multi-channel detector to every full
resolution frame.  It exports permissive proposals; downstream multi-scan
association remains responsible for identity assignment.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ._full_stack_io import (
    crop_box,
    group_frame,
    group_id,
    image_index,
    load_gray,
    prediction_files,
    read_rows,
    write_json,
    write_rows,
)
from ._records import Detection
from .scene_stabilization import StabilizationConfig, phase_translation, translate_image
from .tiny_p2_detector import _tensor_loss, _torch, decode
from .track_conditioned_proposals import RoiConfig, predict_roi


@dataclass(frozen=True)
class TemporalP2Config:
    offsets: tuple[int, ...] = (-2, -1, 1, 2)
    patch_size: int = 192
    min_neighbours: int = 2
    score_threshold: float = 0.003
    top_k: int = 60
    max_training_samples: int = 20_000
    patch_jitter: float = 0.18
    registration: StabilizationConfig = StabilizationConfig(
        max_shift=24.0,
        min_peak_ratio=1.06,
        mask_scale=1.0,
        downsample=1,
    )
    roi: RoiConfig = RoiConfig(
        history=5,
        sigma_scale=3.0,
        min_margin=18.0,
        box_scale=4.0,
        upscale=1.0,
        max_roi_side=256,
        max_track_gap=18,
    )


@dataclass(frozen=True)
class TemporalPatchSample:
    sequence: str
    frame_id: int
    x1: float
    y1: float
    width: float
    height: float
    boxes: tuple[tuple[float, float, float, float], ...]


class _ImageCache:
    def __init__(self, limit: int = 18):
        self.limit = max(2, int(limit))
        self._values: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        if path in self._values:
            value = self._values.pop(path)
            self._values[path] = value
            return value
        value = load_gray(path)
        self._values[path] = value
        while len(self._values) > self.limit:
            self._values.popitem(last=False)
        return value


def _resize(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.shape == (height, width):
        return np.asarray(array, dtype=np.float32)
    image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0))
    return np.asarray(
        image.resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0


def temporal_feature_stack(
    current: np.ndarray,
    neighbours: list[np.ndarray],
    config: TemporalP2Config = TemporalP2Config(),
) -> tuple[np.ndarray, dict]:
    """Build five registered channels for a middle-frame detector input."""

    current = np.asarray(current, dtype=np.float32)
    aligned: list[np.ndarray] = []
    estimates = []
    for neighbour in neighbours:
        height = min(current.shape[0], neighbour.shape[0])
        width = min(current.shape[1], neighbour.shape[1])
        reference = current[:height, :width]
        moving = np.asarray(neighbour[:height, :width], dtype=np.float32)
        estimate = phase_translation(reference, moving, config.registration)
        estimates.append(estimate)
        if estimate.accepted:
            aligned.append(translate_image(moving, estimate.dy, estimate.dx))

    diagnostics = {
        "neighbour_count": len(neighbours),
        "accepted_registrations": len(aligned),
        "peak_ratios": [estimate.peak_ratio for estimate in estimates],
    }
    if len(aligned) < config.min_neighbours:
        aligned.extend(
            current.copy()
            for _ in range(config.min_neighbours - len(aligned))
        )
        diagnostics["fallback"] = True
    else:
        diagnostics["fallback"] = False

    height = min([current.shape[0], *[image.shape[0] for image in aligned]])
    width = min([current.shape[1], *[image.shape[1] for image in aligned]])
    centre = current[:height, :width]
    stack = np.stack([image[:height, :width] for image in aligned])
    background = np.median(stack, axis=0)
    signed = centre - background
    absolute = np.abs(signed)
    temporal_mad = np.median(np.abs(stack - background), axis=0)

    # Keep every channel in a comparable bounded range.  The residual channels
    # are amplified because a 2--8 pixel UAV otherwise occupies very little of
    # the input dynamic range.
    features = np.stack(
        [
            centre,
            background,
            np.clip(0.5 + 2.0 * signed, 0.0, 1.0),
            np.clip(4.0 * absolute, 0.0, 1.0),
            np.clip(4.0 * temporal_mad, 0.0, 1.0),
        ]
    ).astype(np.float32, copy=False)
    return features, diagnostics


def TemporalP2Detector(*, channels: int = 64, blocks: int = 4):
    """Construct the five-channel stride-4 specialist lazily."""

    torch, nn = _torch()

    class Residual(nn.Module):
        def __init__(self, channel_count: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channel_count, channel_count, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_count),
                nn.SiLU(),
                nn.Conv2d(channel_count, channel_count, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_count),
            )

        def forward(self, inputs):
            return torch.nn.functional.silu(inputs + self.net(inputs))

    class Model(nn.Module):
        stride = 4

        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(5, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
                *[Residual(channels) for _ in range(blocks)],
            )
            self.heat = nn.Conv2d(channels, 1, 1)
            self.offset = nn.Conv2d(channels, 2, 1)
            self.log_size = nn.Conv2d(channels, 2, 1)
            nn.init.constant_(self.heat.bias, -4.0)

        def forward(self, inputs):
            features = self.stem(inputs)
            return self.heat(features), self.offset(features), self.log_size(features)

    return Model()


def _patch_bounds(
    row: Detection,
    image_shape: tuple[int, int],
    patch_size: int,
    jitter_x: float,
    jitter_y: float,
) -> tuple[float, float, float, float]:
    height, width = image_shape
    side = min(float(patch_size), float(max(height, width)))
    centre_x = row.center_x + jitter_x * side
    centre_y = row.center_y + jitter_y * side
    x1 = float(np.clip(centre_x - 0.5 * side, 0.0, max(width - side, 0.0)))
    y1 = float(np.clip(centre_y - 0.5 * side, 0.0, max(height - side, 0.0)))
    return x1, y1, min(side, width - x1), min(side, height - y1)


def samples_from_lts(
    image_root: Path,
    label_root: Path,
    sequence_names: Iterable[str] | None = None,
    config: TemporalP2Config = TemporalP2Config(),
    *,
    seed: int = 0,
) -> list[TemporalPatchSample]:
    allowed = None if sequence_names is None else set(sequence_names)
    rng = np.random.default_rng(seed)
    samples: list[TemporalPatchSample] = []
    for label_file in sorted(label_root.glob("*.txt")):
        name = label_file.stem
        if allowed is not None and name not in allowed:
            continue
        frame_paths = image_index(image_root / name)
        if not frame_paths:
            continue
        rows = read_rows(label_file)
        by_frame = group_frame(rows)
        with Image.open(frame_paths[min(frame_paths)]) as image:
            image_shape = (image.height, image.width)
        for frame, frame_rows in sorted(by_frame.items()):
            if frame not in frame_paths:
                continue
            for anchor in frame_rows:
                jitter_x = float(rng.uniform(-config.patch_jitter, config.patch_jitter))
                jitter_y = float(rng.uniform(-config.patch_jitter, config.patch_jitter))
                x1, y1, width, height = _patch_bounds(
                    anchor,
                    image_shape,
                    config.patch_size,
                    jitter_x,
                    jitter_y,
                )
                boxes = []
                for row in frame_rows:
                    if not (x1 <= row.center_x < x1 + width and y1 <= row.center_y < y1 + height):
                        continue
                    boxes.append(
                        (
                            row.center_x - x1,
                            row.center_y - y1,
                            row.width,
                            row.height,
                        )
                    )
                samples.append(
                    TemporalPatchSample(
                        name,
                        frame,
                        x1,
                        y1,
                        width,
                        height,
                        tuple(boxes),
                    )
                )
    if len(samples) > config.max_training_samples:
        indices = rng.choice(len(samples), config.max_training_samples, replace=False)
        samples = [samples[int(index)] for index in sorted(indices)]
    return samples


def _sample_tensor(
    sample: TemporalPatchSample,
    frame_paths: dict[int, Path],
    cache: _ImageCache,
    config: TemporalP2Config,
):
    torch, _ = _torch()
    current, _ = crop_box(
        cache.get(frame_paths[sample.frame_id]),
        sample.x1,
        sample.y1,
        sample.x1 + sample.width,
        sample.y1 + sample.height,
    )
    neighbours = []
    for offset in config.offsets:
        frame = sample.frame_id + offset
        if frame not in frame_paths:
            continue
        crop, _ = crop_box(
            cache.get(frame_paths[frame]),
            sample.x1,
            sample.y1,
            sample.x1 + sample.width,
            sample.y1 + sample.height,
        )
        neighbours.append(crop)
    features, diagnostics = temporal_feature_stack(current, neighbours, config)
    target_size = config.patch_size
    resized = np.stack(
        [_resize(channel, target_size, target_size) for channel in features]
    )
    scale_x = target_size / max(sample.width, 1e-6)
    scale_y = target_size / max(sample.height, 1e-6)
    boxes = np.asarray(
        [
            (center_x * scale_x, center_y * scale_y, width * scale_x, height * scale_y)
            for center_x, center_y, width, height in sample.boxes
        ],
        dtype=np.float32,
    ).reshape(-1, 4)
    return torch.from_numpy(resized[None]), boxes, diagnostics


def train_temporal_detector(
    image_root: Path,
    label_root: Path,
    output: Path,
    sequence_names: list[str] | None = None,
    *,
    epochs: int = 4,
    learning_rate: float = 0.0015,
    seed: int = 0,
    device: str = "cpu",
    config: TemporalP2Config = TemporalP2Config(),
) -> dict:
    torch, _ = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    samples = samples_from_lts(
        image_root,
        label_root,
        sequence_names,
        config,
        seed=seed,
    )
    if not samples:
        raise ValueError("no temporal P2 training samples")
    frame_indices = {
        name: image_index(image_root / name) for name in sorted({sample.sequence for sample in samples})
    }
    model = TemporalP2Detector().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0001,
    )
    cache = _ImageCache()
    history = []
    accepted = attempted = 0
    for _ in range(max(1, epochs)):
        random.shuffle(samples)
        losses = []
        model.train()
        for sample in samples:
            tensor, boxes, diagnostics = _sample_tensor(
                sample,
                frame_indices[sample.sequence],
                cache,
                config,
            )
            accepted += int(diagnostics["accepted_registrations"])
            attempted += int(diagnostics["neighbour_count"])
            tensor = tensor.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _tensor_loss(model(tensor), [boxes])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stride": 4,
            "history": history,
            "format": "raft-uav-temporal-p2-v1",
            "config": {
                "offsets": list(config.offsets),
                "patch_size": config.patch_size,
            },
        },
        output,
    )
    return {
        "sample_count": len(samples),
        "epochs": max(1, epochs),
        "final_loss": history[-1],
        "checkpoint": str(output),
        "accepted_registrations": accepted,
        "attempted_registrations": attempted,
    }


def load_temporal_detector(checkpoint: Path, device: str = "cpu"):
    torch, _ = _torch()
    payload = torch.load(checkpoint, map_location=device)
    if payload.get("format") != "raft-uav-temporal-p2-v1":
        raise ValueError(f"unsupported temporal P2 checkpoint: {checkpoint}")
    model = TemporalP2Detector().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def generate_temporal_p2_roi(
    checkpoint: Path,
    image_root: Path,
    tracks_dir: Path,
    output_dir: Path,
    sequence_names: list[str] | None = None,
    *,
    device: str = "cpu",
    config: TemporalP2Config = TemporalP2Config(),
) -> dict:
    torch, _ = _torch()
    model = load_temporal_detector(checkpoint, device)
    allowed = None if sequence_names is None else set(sequence_names)
    summary: dict[str, dict] = {}
    for track_file in prediction_files(tracks_dir):
        name = track_file.stem
        if allowed is not None and name not in allowed:
            continue
        frame_paths = image_index(image_root / name)
        tracks = group_id(read_rows(track_file))
        cache = _ImageCache()
        proposals: list[Detection] = []
        accepted = attempted = roi_count = 0
        for frame, current_path in sorted(frame_paths.items()):
            current_image = cache.get(current_path)
            local: list[tuple[float, float, float, float, float]] = []
            for history in tracks.values():
                try:
                    x1, y1, width, height = predict_roi(
                        history,
                        frame,
                        current_image.shape,
                        config.roi,
                    )
                except ValueError:
                    continue
                roi_count += 1
                current, bounds = crop_box(
                    current_image,
                    x1,
                    y1,
                    x1 + width,
                    y1 + height,
                )
                neighbours = []
                for offset in config.offsets:
                    other = frame + offset
                    if other not in frame_paths:
                        continue
                    crop, _ = crop_box(
                        cache.get(frame_paths[other]),
                        x1,
                        y1,
                        x1 + width,
                        y1 + height,
                    )
                    neighbours.append(crop)
                features, diagnostics = temporal_feature_stack(current, neighbours, config)
                accepted += int(diagnostics["accepted_registrations"])
                attempted += int(diagnostics["neighbour_count"])
                resized = np.stack(
                    [
                        _resize(channel, config.patch_size, config.patch_size)
                        for channel in features
                    ]
                )
                tensor = torch.from_numpy(resized[None]).to(device)
                with torch.no_grad():
                    boxes = decode(
                        model(tensor),
                        (config.patch_size, config.patch_size),
                        config.score_threshold,
                        config.top_k,
                    )
                scale_x = width / config.patch_size
                scale_y = height / config.patch_size
                for local_x, local_y, local_width, local_height, score in boxes:
                    local.append(
                        (
                            bounds[0] + local_x * scale_x,
                            bounds[1] + local_y * scale_y,
                            max(1.0, local_width * scale_x),
                            max(1.0, local_height * scale_y),
                            score,
                        )
                    )
            seen: set[tuple[float, float, float, float]] = set()
            for x1, y1, width, height, score in sorted(local, key=lambda row: -row[4]):
                key = tuple(round(value, 1) for value in (x1, y1, width, height))
                if key in seen:
                    continue
                seen.add(key)
                proposals.append(
                    Detection(
                        frame,
                        len(proposals) + 1,
                        x1,
                        y1,
                        width,
                        height,
                        score,
                        1,
                        1.0,
                    )
                )
        write_rows(output_dir / f"{name}.txt", proposals)
        summary[name] = {
            "proposal_count": len(proposals),
            "roi_count": roi_count,
            "accepted_registrations": accepted,
            "attempted_registrations": attempted,
        }
    return {
        "sequences": summary,
        "total_proposals": sum(value["proposal_count"] for value in summary.values()),
        "config": {
            "offsets": list(config.offsets),
            "patch_size": config.patch_size,
            "score_threshold": config.score_threshold,
        },
    }


def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--image-root", type=Path, required=True)
    train.add_argument("--label-root", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--sequence-list", type=Path)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--max-training-samples", type=int, default=20_000)
    train.add_argument("--device", default="cpu")
    train.add_argument("--summary-json", type=Path)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--image-root", type=Path, required=True)
    predict.add_argument("--tracks-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--sequence-list", type=Path)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--score-threshold", type=float, default=0.003)
    predict.add_argument("--summary-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        result = train_temporal_detector(
            args.image_root,
            args.label_root,
            args.output,
            _names(args.sequence_list),
            epochs=args.epochs,
            device=args.device,
            config=TemporalP2Config(max_training_samples=args.max_training_samples),
        )
    else:
        result = generate_temporal_p2_roi(
            args.checkpoint,
            args.image_root,
            args.tracks_dir,
            args.output_dir,
            _names(args.sequence_list),
            device=args.device,
            config=TemporalP2Config(score_threshold=args.score_threshold),
        )
    if args.summary_json:
        write_json(args.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
