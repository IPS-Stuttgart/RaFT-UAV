"""I/O and image helpers shared by the full-stack LTS experiments."""
from __future__ import annotations
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image
from ._records import Detection, format_detection, parse_detection_text
_IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}

def prediction_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted((p for p in path.glob('*.txt') if p.is_file()))

def read_rows(path: Path) -> list[Detection]:
    return parse_detection_text(path.read_text(encoding='utf-8'), source=str(path))

def write_rows(path: Path, rows: Iterable[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r.frame_id, r.object_id, -r.confidence, r.x1, r.y1))
    text = '\n'.join((format_detection(r) for r in ordered))
    path.write_text(text + ('\n' if text else ''), encoding='utf-8')

def group_frame(rows: Iterable[Detection]) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    for r in rows:
        out.setdefault(r.frame_id, []).append(r)
    return out

def group_id(rows: Iterable[Detection]) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    for r in rows:
        out.setdefault(r.object_id, []).append(r)
    for values in out.values():
        values.sort(key=lambda r: r.frame_id)
    return out

def image_index(sequence_dir: Path) -> dict[int, Path]:
    files = sorted((p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES))
    out = {}
    for ordinal, p in enumerate(files, 1):
        nums = re.findall('\\d+', p.stem)
        frame = int(nums[-1]) if nums else ordinal
        if frame in out:
            frame = ordinal
        out[frame] = p
    return out

def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('L'), dtype=np.float32) / 255.0

def save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).save(path)

def clip_box(row: Detection, width: int, height: int) -> Detection:
    x1 = float(np.clip(row.x1, 0, max(width - 1, 0)))
    y1 = float(np.clip(row.y1, 0, max(height - 1, 0)))
    x2 = float(np.clip(row.x1 + row.width, x1 + 1, width))
    y2 = float(np.clip(row.y1 + row.height, y1 + 1, height))
    return replace(row, x1=x1, y1=y1, width=x2 - x1, height=y2 - y1)

def crop_box(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    xa = max(0, int(np.floor(x1)))
    ya = max(0, int(np.floor(y1)))
    xb = min(w, max(xa + 1, int(np.ceil(x2))))
    yb = min(h, max(ya + 1, int(np.ceil(y2))))
    return (image[ya:yb, xa:xb], (xa, ya, xb, yb))

def iou(a: Detection, b: Detection) -> float:
    ax2 = a.x1 + a.width
    ay2 = a.y1 + a.height
    bx2 = b.x1 + b.width
    by2 = b.y1 + b.height
    iw = max(0, min(ax2, bx2) - max(a.x1, b.x1))
    ih = max(0, min(ay2, by2) - max(a.y1, b.y1))
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return 0.0 if union <= 0 else inter / union

def fuse_prediction_dirs(sources: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = sorted({p.name for src in sources for p in prediction_files(src)})
    for name in names:
        rows = []
        next_id = 1
        for src in sources:
            file = src / name
            if not file.is_file():
                continue
            for row in read_rows(file):
                rows.append(replace(row, object_id=next_id))
                next_id += 1
        write_rows(output / name, rows)

def copy_selected(source: Path, destination: Path, names: Iterable[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = source / f'{name}.txt'
        if src.is_file():
            shutil.copy2(src, destination / src.name)

def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
