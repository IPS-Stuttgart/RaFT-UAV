"""Sequence-level image evidence for MMUAD classification and ranking."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.sequence import discover_sequence_paths, official_track5_sequence_timestamps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_DIR_NAMES = {"image", "images", "camera", "cameras"}
IMAGE_EVIDENCE_MODE = "sequence-level-no-calibration"
IMAGE_FEATURE_BACKENDS = (
    "handcrafted",
    "auto",
    "torchvision-resnet18",
    "torchvision-efficientnet-b0",
)
IMAGE_FILE_ROW_COLUMNS = ["image_path", "image_time_s"]


@dataclass(frozen=True)
class ImageEvidenceResult:
    """Frame- and sequence-level image evidence tables."""

    sequence_features: pd.DataFrame
    frame_features: pd.DataFrame


def build_image_evidence(
    sequence_root: Path,
    *,
    truth_file: Path | None = None,
    sequence_glob: str = "*",
    timestamp_source: str = "image",
    max_frames_per_sequence: int = 32,
    max_image_time_delta_s: float | None = 0.5,
    image_feature_backend: str = "handcrafted",
) -> ImageEvidenceResult:
    """Sample official image frames and extract conservative visual evidence.

    This deliberately emits sequence-level evidence only.  It does not project
    3D candidates into pixels unless a future calibrated projection path is
    added.
    """

    backend_requested = _normalize_image_feature_backend(image_feature_backend)
    backend_resolved, feature_extractor = _make_image_feature_extractor(backend_requested)
    sequences = discover_sequence_paths(Path(sequence_root), sequence_glob=sequence_glob)
    truth_by_sequence = _truth_times_by_sequence(truth_file)
    frame_records: list[dict[str, Any]] = []
    for paths in sequences:
        image_files = _sequence_image_files(paths.root)
        if not image_files:
            continue
        image_rows = _image_file_rows(image_files)
        if image_rows.empty:
            continue
        target_times = truth_by_sequence.get(paths.sequence_id)
        if target_times is None:
            try:
                target_times = official_track5_sequence_timestamps(
                    paths,
                    timestamp_source=timestamp_source,
                )
            except ValueError:
                target_times = []
        if not target_times:
            target_times = image_rows["image_time_s"].dropna().astype(float).tolist()
        for target_time_s, image_row in _sample_nearest_image_rows(
            image_rows,
            target_times,
            max_frames=max_frames_per_sequence,
            max_time_delta_s=max_image_time_delta_s,
        ):
            record = feature_extractor(Path(image_row["image_path"]))
            record.update(
                {
                    "sequence_id": paths.sequence_id,
                    "target_time_s": float(target_time_s),
                    "image_time_s": float(image_row["image_time_s"]),
                    "image_time_delta_s": float(image_row["image_time_s"] - target_time_s),
                    "image_path": str(image_row["image_path"]),
                    "image_evidence_mode": IMAGE_EVIDENCE_MODE,
                    "image_feature_backend_requested": backend_requested,
                    "image_feature_backend_resolved": backend_resolved,
                }
            )
            frame_records.append(record)
    frame_features = pd.DataFrame.from_records(frame_records)
    sequence_features = _sequence_features_from_frame_features(
        frame_features,
        target_counts={
            paths.sequence_id: len(truth_by_sequence.get(paths.sequence_id, []))
            for paths in sequences
        },
    )
    return ImageEvidenceResult(
        sequence_features=sequence_features,
        frame_features=frame_features,
    )


def write_image_evidence(result: ImageEvidenceResult, output_dir: Path) -> dict[str, str]:
    """Write image evidence artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_csv = output_dir / "mmuad_image_evidence.csv"
    frame_csv = output_dir / "mmuad_image_frame_evidence.csv"
    result.sequence_features.to_csv(sequence_csv, index=False)
    result.frame_features.to_csv(frame_csv, index=False)
    return {
        "image_evidence_csv": str(sequence_csv),
        "image_frame_evidence_csv": str(frame_csv),
    }


def _truth_times_by_sequence(truth_file: Path | None) -> dict[str, list[float]]:
    if truth_file is None:
        return {}
    rows = load_evaluation_truth_file(Path(truth_file)).rows
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["time_s"] = pd.to_numeric(rows["time_s"], errors="coerce")
    out: dict[str, list[float]] = {}
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        times = group["time_s"].dropna().astype(float).to_numpy()
        out[str(sequence_id)] = sorted(float(value) for value in np.unique(times))
    return out


def _sequence_image_files(root: Path) -> list[Path]:
    files: list[Path] = []
    root = Path(root)
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not directory.is_dir():
            continue
        normalized = directory.name.lower().replace("-", "_").replace(" ", "_")
        if normalized in IMAGE_DIR_NAMES:
            files.extend(_image_files_under(directory))
    if files:
        return sorted(set(files))
    return _image_files_under(root)


def _image_files_under(path: Path) -> list[Path]:
    return [
        item
        for item in sorted(Path(path).rglob("*"))
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]


def _image_file_rows(image_files: list[Path]) -> pd.DataFrame:
    records = []
    for path in image_files:
        timestamp = _timestamp_from_filename(path)
        if timestamp is None:
            continue
        records.append({"image_path": str(path), "image_time_s": float(timestamp)})
    return (
        pd.DataFrame.from_records(records, columns=IMAGE_FILE_ROW_COLUMNS)
        .sort_values("image_time_s")
        .reset_index(drop=True)
    )


def _sample_nearest_image_rows(
    image_rows: pd.DataFrame,
    target_times: list[float],
    *,
    max_frames: int,
    max_time_delta_s: float | None,
):
    target_times = sorted(float(value) for value in target_times if np.isfinite(float(value)))
    if max_frames > 0 and len(target_times) > max_frames:
        indices = np.linspace(0, len(target_times) - 1, int(max_frames)).round().astype(int)
        target_times = [target_times[int(index)] for index in np.unique(indices)]
    image_times = image_rows["image_time_s"].to_numpy(float)
    emitted_paths: set[str] = set()
    for target_time in target_times:
        nearest_idx = int(np.argmin(np.abs(image_times - target_time)))
        delta = abs(float(image_times[nearest_idx] - target_time))
        if max_time_delta_s is not None and delta > float(max_time_delta_s):
            continue
        row = image_rows.iloc[nearest_idx]
        path = str(row["image_path"])
        if path in emitted_paths:
            continue
        emitted_paths.add(path)
        yield target_time, row


def _normalize_image_feature_backend(backend: str) -> str:
    normalized = str(backend).strip().lower().replace("_", "-")
    aliases = {
        "resnet18": "torchvision-resnet18",
        "efficientnet-b0": "torchvision-efficientnet-b0",
        "efficientnet": "torchvision-efficientnet-b0",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in IMAGE_FEATURE_BACKENDS:
        allowed = ", ".join(IMAGE_FEATURE_BACKENDS)
        raise ValueError(f"unsupported image feature backend {backend!r}; allowed={allowed}")
    return normalized


def _make_image_feature_extractor(backend: str):
    if backend == "handcrafted":
        return "handcrafted", _handcrafted_image_feature_record
    if backend == "auto":
        try:
            return _make_torchvision_feature_extractor("torchvision-resnet18")
        except Exception:
            return "handcrafted", _handcrafted_image_feature_record
    return _make_torchvision_feature_extractor(backend)


def _make_torchvision_feature_extractor(backend: str):
    try:
        import torch
        from PIL import Image
        from torchvision import models
    except Exception as exc:  # pragma: no cover - depends on optional deep stack
        raise ValueError(
            f"{backend} image features require torch, torchvision, and pillow"
        ) from exc

    if backend == "torchvision-resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        feature_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif backend == "torchvision-efficientnet-b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        feature_model = torch.nn.Sequential(model.features, model.avgpool)
    else:  # pragma: no cover - caller normalizes backends
        raise ValueError(f"unsupported torchvision image backend {backend!r}")
    preprocess = weights.transforms()
    feature_model.eval()

    def extractor(path: Path) -> dict[str, Any]:
        record = _handcrafted_image_feature_record(path)
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            embedding = feature_model(tensor).flatten().detach().cpu().numpy().astype(float)
        for idx, value in enumerate(embedding):
            record[f"image_pretrained_embedding_{idx}"] = float(value)
        return record

    return backend, extractor


def _handcrafted_image_feature_record(path: Path) -> dict[str, Any]:
    image = _read_image_rgb(path)
    height, width = image.shape[:2]
    pixels = _sample_pixels(image)
    luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    saturation = _saturation(pixels)
    edge_score = _edge_score(image)
    luma_hist, _ = np.histogram(luma, bins=8, range=(0.0, 1.0), density=False)
    luma_hist = luma_hist.astype(float) / max(float(luma_hist.sum()), 1.0)
    objectness = _objectness_score(luma=luma, saturation=saturation, edge_score=edge_score)
    record: dict[str, Any] = {
        "image_width_px": int(width),
        "image_height_px": int(height),
        "image_luma_mean": float(np.mean(luma)),
        "image_luma_std": float(np.std(luma)),
        "image_saturation_mean": float(np.mean(saturation)),
        "image_edge_score": float(edge_score),
        "image_dark_fraction": float(np.mean(luma < 0.15)),
        "image_bright_fraction": float(np.mean(luma > 0.85)),
        "image_objectness_score": float(objectness),
        "image_embedding_r_mean": float(np.mean(pixels[:, 0])),
        "image_embedding_g_mean": float(np.mean(pixels[:, 1])),
        "image_embedding_b_mean": float(np.mean(pixels[:, 2])),
    }
    for idx, value in enumerate(luma_hist):
        record[f"image_embedding_luma_bin_{idx}"] = float(value)
    _add_rgb_histogram_features(record, pixels)
    _add_center_crop_features(record, image)
    _add_quadrant_features(record, image)
    _add_gradient_orientation_features(record, image)
    return record


def _read_image_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore[assignment]
    if Image is not None:
        with Image.open(path) as image_file:
            array = np.asarray(image_file.convert("RGB"), dtype=float) / 255.0
        return np.clip(array, 0.0, 1.0)
    try:
        from matplotlib import image as mpimg
    except Exception as exc:  # pragma: no cover - matplotlib is a project dependency
        raise ValueError("image evidence requires matplotlib image reading support") from exc
    array = np.asarray(mpimg.imread(path))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"unsupported image shape for {path}: {array.shape}")
    array = array[:, :, :3].astype(float)
    if array.max(initial=0.0) > 1.0:
        array /= 255.0
    return np.clip(array, 0.0, 1.0)


def _sample_pixels(image: np.ndarray, *, max_pixels: int = 16384) -> np.ndarray:
    pixels = image.reshape(-1, 3)
    if len(pixels) <= max_pixels:
        return pixels
    step = int(np.ceil(len(pixels) / max_pixels))
    return pixels[::step]


def _saturation(pixels: np.ndarray) -> np.ndarray:
    max_channel = np.max(pixels, axis=1)
    min_channel = np.min(pixels, axis=1)
    return (max_channel - min_channel) / np.maximum(max_channel, 1.0e-6)


def _edge_score(image: np.ndarray) -> float:
    gray = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    if min(gray.shape) < 2:
        return 0.0
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    return float(0.5 * (np.mean(dx) + np.mean(dy)))


def _objectness_score(*, luma: np.ndarray, saturation: np.ndarray, edge_score: float) -> float:
    contrast = float(np.std(luma))
    saliency = 0.45 * np.clip(contrast * 4.0, 0.0, 1.0)
    saliency += 0.35 * np.clip(edge_score * 8.0, 0.0, 1.0)
    saliency += 0.20 * np.clip(float(np.mean(saturation)) * 2.0, 0.0, 1.0)
    return float(np.clip(saliency, 0.0, 1.0))


def _add_rgb_histogram_features(record: dict[str, Any], pixels: np.ndarray) -> None:
    for channel_idx, channel in enumerate(("r", "g", "b")):
        hist, _ = np.histogram(
            pixels[:, channel_idx],
            bins=8,
            range=(0.0, 1.0),
            density=False,
        )
        hist = hist.astype(float) / max(float(hist.sum()), 1.0)
        for idx, value in enumerate(hist):
            record[f"image_embedding_{channel}_bin_{idx}"] = float(value)


def _add_center_crop_features(record: dict[str, Any], image: np.ndarray) -> None:
    height, width = image.shape[:2]
    crop_height = max(1, int(round(height * 0.5)))
    crop_width = max(1, int(round(width * 0.5)))
    y0 = max(0, (height - crop_height) // 2)
    x0 = max(0, (width - crop_width) // 2)
    crop = image[y0 : y0 + crop_height, x0 : x0 + crop_width]
    pixels = _sample_pixels(crop)
    luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    saturation = _saturation(pixels)
    edge_score = _edge_score(crop)
    record["image_center_luma_mean"] = float(np.mean(luma))
    record["image_center_luma_std"] = float(np.std(luma))
    record["image_center_saturation_mean"] = float(np.mean(saturation))
    record["image_center_edge_score"] = float(edge_score)
    record["image_center_dark_fraction"] = float(np.mean(luma < 0.15))
    record["image_center_bright_fraction"] = float(np.mean(luma > 0.85))
    record["image_center_objectness_score"] = float(
        _objectness_score(luma=luma, saturation=saturation, edge_score=edge_score)
    )


def _add_quadrant_features(record: dict[str, Any], image: np.ndarray) -> None:
    height, width = image.shape[:2]
    y_mid = max(1, height // 2)
    x_mid = max(1, width // 2)
    quadrants = {
        "tl": image[:y_mid, :x_mid],
        "tr": image[:y_mid, x_mid:],
        "bl": image[y_mid:, :x_mid],
        "br": image[y_mid:, x_mid:],
    }
    for name, quadrant in quadrants.items():
        if quadrant.size == 0:
            continue
        pixels = _sample_pixels(quadrant)
        luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
        record[f"image_quadrant_{name}_luma_mean"] = float(np.mean(luma))
        record[f"image_quadrant_{name}_luma_std"] = float(np.std(luma))
        record[f"image_quadrant_{name}_saturation_mean"] = float(np.mean(_saturation(pixels)))


def _add_gradient_orientation_features(record: dict[str, Any], image: np.ndarray) -> None:
    gray = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    if min(gray.shape) < 3:
        for idx in range(8):
            record[f"image_grad_orientation_bin_{idx}"] = 0.0
        return
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(
        orientation.reshape(-1),
        bins=8,
        range=(0.0, float(np.pi)),
        weights=magnitude.reshape(-1),
    )
    hist = hist.astype(float) / max(float(hist.sum()), 1.0e-12)
    for idx, value in enumerate(hist):
        record[f"image_grad_orientation_bin_{idx}"] = float(value)


def _sequence_features_from_frame_features(
    frame_features: pd.DataFrame,
    *,
    target_counts: dict[str, int],
) -> pd.DataFrame:
    if frame_features.empty:
        return pd.DataFrame(columns=["sequence_id"])
    records: list[dict[str, Any]] = []
    for sequence_id, group in frame_features.groupby("sequence_id", sort=True):
        record: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "image_evidence_mode": IMAGE_EVIDENCE_MODE,
            "image_feature_backend_resolved": ";".join(
                sorted(set(group.get("image_feature_backend_resolved", pd.Series(dtype=str)).astype(str)))
            ),
            "image_sampled_frame_count": int(len(group)),
            "image_target_count": int(target_counts.get(str(sequence_id), len(group))),
        }
        record["image_matched_target_fraction"] = float(
            len(group) / max(int(record["image_target_count"]), 1)
        )
        for column in _numeric_image_feature_columns(group):
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            if values.size == 0:
                continue
            if column.startswith("image_embedding_luma_bin_"):
                record[column] = float(np.mean(values))
                continue
            record[f"{column}_mean"] = float(np.mean(values))
            record[f"{column}_std"] = float(np.std(values))
            record[f"{column}_min"] = float(np.min(values))
            record[f"{column}_max"] = float(np.max(values))
            record[f"{column}_p90"] = float(np.percentile(values, 90.0))
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values("sequence_id").reset_index(drop=True)


def _numeric_image_feature_columns(rows: pd.DataFrame) -> list[str]:
    skip = {"sequence_id", "target_time_s", "image_time_s"}
    columns: list[str] = []
    for column in rows.columns:
        if column in skip or not str(column).startswith("image_"):
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        if values.notna().any():
            columns.append(str(column))
    return columns


def _timestamp_from_filename(path: Path) -> float | None:
    tokens = re.findall(r"[-+]?\d*\.?\d+", Path(path).stem)
    if not tokens:
        return None
    try:
        return float(tokens[-1])
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-image-evidence",
        description="extract conservative sequence-level image evidence for MMUAD",
    )
    parser.add_argument("sequence_root", type=Path)
    parser.add_argument("--truth-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--timestamp-source", default="image")
    parser.add_argument("--max-frames-per-sequence", type=int, default=32)
    parser.add_argument("--max-image-time-delta-s", type=float, default=0.5)
    parser.add_argument(
        "--image-feature-backend",
        choices=IMAGE_FEATURE_BACKENDS,
        default="handcrafted",
        help=(
            "visual feature extractor; torchvision backends emit pretrained embeddings "
            "when torch/torchvision are installed"
        ),
    )
    args = parser.parse_args(argv)

    result = build_image_evidence(
        args.sequence_root,
        truth_file=args.truth_file,
        sequence_glob=args.sequence_glob,
        timestamp_source=args.timestamp_source,
        max_frames_per_sequence=args.max_frames_per_sequence,
        max_image_time_delta_s=args.max_image_time_delta_s,
        image_feature_backend=args.image_feature_backend,
    )
    paths = write_image_evidence(result, args.output_dir)
    print("mmuad_image_evidence=ok")
    for key, value in paths.items():
        print(f"{key}={value}")
    print(f"sequence_rows={len(result.sequence_features)}")
    print(f"frame_rows={len(result.frame_features)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
