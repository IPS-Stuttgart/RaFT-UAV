"""I/O helpers for normalized MMUAD tracking candidates."""

from __future__ import annotations

import gzip
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import (
    CandidateFrame,
    TruthFrame,
    normalize_candidate_columns,
    normalize_truth_columns,
)

DELIMITED_TABLE_SUFFIXES = {".csv", ".tsv", ".txt"}
JSON_TABLE_SUFFIXES = {".json", ".jsonl", ".ndjson"}
COMPRESSED_TABLE_SUFFIX = ".gz"
DISCOVERABLE_DELIMITED_TABLE_SUFFIXES = tuple(
    sorted(
        DELIMITED_TABLE_SUFFIXES
        | {f"{suffix}{COMPRESSED_TABLE_SUFFIX}" for suffix in DELIMITED_TABLE_SUFFIXES}
    )
)
DISCOVERABLE_JSON_TABLE_SUFFIXES = tuple(
    sorted(
        JSON_TABLE_SUFFIXES
        | {f"{suffix}{COMPRESSED_TABLE_SUFFIX}" for suffix in JSON_TABLE_SUFFIXES}
    )
)


def data_file_suffix(path: Path) -> str:
    """Return the logical data suffix, treating ``*.gz`` as transparent compression."""

    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    if not suffixes:
        return ""
    if suffixes[-1] == COMPRESSED_TABLE_SUFFIX and len(suffixes) >= 2:
        return suffixes[-2]
    return suffixes[-1]


def path_matches_suffix(path: Path, suffixes: Iterable[str]) -> bool:
    """Return true when ``path`` ends in any literal suffix from ``suffixes``."""

    name = Path(path).name.lower()
    return any(name.endswith(suffix.lower()) for suffix in suffixes)


def read_text_export(path: Path, *, errors: str | None = None) -> str:
    """Read UTF-8 text from a plain or gzip-compressed export file."""

    path = Path(path)
    if path.suffix.lower() == COMPRESSED_TABLE_SUFFIX:
        with gzip.open(path, "rt", encoding="utf-8", errors=errors) as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors=errors)


def read_binary_export(path: Path) -> bytes:
    """Read bytes from a plain or gzip-compressed export file."""

    path = Path(path)
    if path.suffix.lower() == COMPRESSED_TABLE_SUFFIX:
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def load_candidate_csv(
    path: Path,
    *,
    default_sequence_id: str = "default",
    source: str = "candidate",
) -> CandidateFrame:
    """Load a normalized or alias-compatible candidate CSV."""

    raw = pd.read_csv(path)
    if not _has_any_column(raw, ("source", "sensor", "modality")):
        raw["source"] = source
    rows = normalize_candidate_columns(
        raw,
        default_sequence_id=default_sequence_id,
    )
    frame = CandidateFrame(rows)
    frame.validate()
    return frame


def load_candidate_file(
    path: Path,
    *,
    default_sequence_id: str = "default",
    source: str = "candidate",
) -> CandidateFrame:
    """Load a normalized candidate table from CSV/TXT/JSON/JSONL or NumPy trajectory rows."""

    path = Path(path)
    suffix = data_file_suffix(path)
    if suffix == ".csv":
        return load_candidate_csv(
            path,
            default_sequence_id=default_sequence_id,
            source=source,
        )
    if suffix in DELIMITED_TABLE_SUFFIXES - {".csv"}:
        raw = _read_delimited_table(path)
    elif suffix in JSON_TABLE_SUFFIXES:
        raw = _read_json_table(
            path,
            preferred=(
                "candidates",
                "detections",
                "tracks",
                "trajectory",
                "trajectories",
                "poses",
                "rows",
                "data",
            ),
        )
    elif suffix in {".npy", ".npz"}:
        raw = _read_numpy_trajectory_table(path)
    else:
        raise ValueError(f"unsupported candidate table extension: {path.suffix}")
    if not _has_any_column(raw, ("source", "sensor", "modality")):
        raw["source"] = source
    rows = normalize_candidate_columns(raw, default_sequence_id=default_sequence_id)
    frame = CandidateFrame(rows)
    frame.validate()
    return frame


def load_truth_csv(path: Path, *, default_sequence_id: str = "default") -> TruthFrame:
    """Load a normalized or alias-compatible truth CSV."""

    rows = normalize_truth_columns(
        pd.read_csv(path),
        default_sequence_id=default_sequence_id,
    )
    frame = TruthFrame(rows)
    frame.validate()
    return frame


def load_truth_file(path: Path, *, default_sequence_id: str = "default") -> TruthFrame:
    """Load a normalized truth table from CSV/TXT/JSON/JSONL or NumPy trajectory rows."""

    path = Path(path)
    suffix = data_file_suffix(path)
    if suffix == ".csv":
        return load_truth_csv(path, default_sequence_id=default_sequence_id)
    if suffix in DELIMITED_TABLE_SUFFIXES - {".csv"}:
        rows = normalize_truth_columns(
            _read_delimited_table(path),
            default_sequence_id=default_sequence_id,
        )
    elif suffix in JSON_TABLE_SUFFIXES:
        rows = normalize_truth_columns(
            _read_json_table(
                path,
                preferred=(
                    "truth",
                    "ground_truth",
                    "gt",
                    "poses",
                    "trajectory",
                    "trajectories",
                    "rows",
                    "data",
                ),
            ),
            default_sequence_id=default_sequence_id,
        )
    elif suffix in {".npy", ".npz"}:
        rows = normalize_truth_columns(
            _read_numpy_trajectory_table(path),
            default_sequence_id=default_sequence_id,
        )
    else:
        raise ValueError(f"unsupported truth table extension: {path.suffix}")
    frame = TruthFrame(rows)
    frame.validate()
    return frame


def load_point_cloud_csv_as_candidates(
    path: Path,
    *,
    source: str = "lidar-cluster",
    voxel_size_m: float = 0.75,
    min_points: int = 3,
    min_confidence: float = 0.0,
) -> CandidateFrame:
    """Cluster simple point-cloud CSV rows into UAV candidate centroids.

    Expected point columns are compatible with the normalized schema aliases:
    ``sequence_id``, ``time_s``, ``x_m``, ``y_m``, ``z_m``.  The clustering is a
    lightweight connected-component pass in voxel space, intended as a first
    baseline and smoke-test feature rather than a competitive MMUAD detector.
    """

    points = normalize_truth_columns(pd.read_csv(path))
    return _point_rows_to_candidates(
        points,
        source=source,
        voxel_size_m=voxel_size_m,
        min_points=min_points,
        min_confidence=min_confidence,
    )




def load_point_cloud_file_as_candidates(
    path: Path,
    *,
    source: str | None = None,
    sequence_id: str | None = None,
    time_s: float | None = None,
    voxel_size_m: float = 0.75,
    min_points: int = 3,
    min_confidence: float = 0.0,
) -> CandidateFrame:
    """Load exported point-cloud files and cluster them into candidates.

    CSV/TSV/TXT/JSON/JSONL, NumPy, PCD, PLY, uncompressed LAS, optional
    LASzip/LAZ via ``laspy``, and simple float32 ``.bin`` files are supported
    as pragmatic exported-data bridges, including gzip-compressed variants.
    This is **not** a native Livox packet reader.  Files without per-point
    timestamps are treated as one frame; ``time_s`` is inferred from the
    filename when it contains a numeric token, otherwise it defaults to ``0.0``.
    """

    path = Path(path)
    suffix = data_file_suffix(path)
    source = source or path.stem.replace("_points", "-cluster")
    if suffix in DELIMITED_TABLE_SUFFIXES:
        points = _read_point_cloud_csv(path)
    elif suffix in JSON_TABLE_SUFFIXES:
        points = _read_point_cloud_json(path)
    elif suffix in {".npy", ".npz"}:
        points = _read_numpy_point_cloud(path)
    elif suffix == ".pcd":
        points = _read_pcd(path)
    elif suffix == ".ply":
        points = _read_ply(path)
    elif suffix == ".las":
        points = _read_las(path)
    elif suffix == ".laz":
        points = _read_las_with_laspy(path)
    elif suffix == ".bin":
        points = _read_binary_point_cloud(path)
    else:
        raise ValueError(f"unsupported point-cloud extension: {path.suffix}")
    points = _add_point_cloud_metadata(
        points,
        path=path,
        sequence_id=sequence_id,
        time_s=time_s,
    )
    return _point_rows_to_candidates(
        points,
        source=source,
        voxel_size_m=voxel_size_m,
        min_points=min_points,
        min_confidence=min_confidence,
    )


def _read_point_cloud_csv(path: Path) -> pd.DataFrame:
    frame = _read_delimited_table(path)
    try:
        return normalize_truth_columns(frame)
    except ValueError as exc:
        if "time_s" not in str(exc):
            raise
    return _normalize_point_frame(frame, path=path)


def _read_point_cloud_json(path: Path) -> pd.DataFrame:
    frame = read_json_table_export(
        path,
        preferred=(
            "points",
            "point_cloud",
            "pointcloud",
            "cloud",
            "lidar_points",
            "livox_points",
            "detections",
            "rows",
            "data",
        ),
    )
    try:
        return normalize_truth_columns(frame)
    except ValueError as exc:
        if "time_s" not in str(exc):
            raise
    return _normalize_point_frame(frame, path=path)


def _add_point_cloud_metadata(
    points: pd.DataFrame,
    *,
    path: Path,
    sequence_id: str | None,
    time_s: float | None,
) -> pd.DataFrame:
    out = points.copy()
    default_sequence_id = str(sequence_id) if sequence_id is not None else path.parent.name
    if sequence_id is not None or "sequence_id" not in out.columns:
        out["sequence_id"] = default_sequence_id
    else:
        out["sequence_id"] = out["sequence_id"].fillna(default_sequence_id).astype(str)

    default_time_s = float(time_s) if time_s is not None else infer_time_s_from_filename(path)
    if time_s is not None or "time_s" not in out.columns:
        out["time_s"] = default_time_s
    else:
        out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce").fillna(default_time_s)
    return out


def infer_time_s_from_filename(path: Path) -> float:
    """Infer a frame timestamp from the last numeric token in a filename."""

    import re

    tokens = re.findall(r"[-+]?\d*\.?\d+", Path(path).stem)
    if not tokens:
        return 0.0
    return float(tokens[-1])


def point_rows_to_candidates(
    points: pd.DataFrame,
    *,
    source: str = "lidar-cluster",
    voxel_size_m: float = 0.75,
    min_points: int = 3,
    min_confidence: float = 0.0,
) -> CandidateFrame:
    """Cluster normalized point rows into candidate centroids.

    This public wrapper is shared by CSV/PCD/PLY and ROS PointCloud2 bridges.
    """

    normalized = normalize_truth_columns(points)
    return _point_rows_to_candidates(
        normalized,
        source=source,
        voxel_size_m=voxel_size_m,
        min_points=min_points,
        min_confidence=min_confidence,
    )


def merge_candidate_frames(frames: Iterable[CandidateFrame]) -> CandidateFrame:
    """Merge several candidate frames, preserving normalization."""

    rows = [frame.rows for frame in frames if not frame.rows.empty]
    if not rows:
        return CandidateFrame(normalize_candidate_columns(pd.DataFrame()))
    return CandidateFrame(normalize_candidate_columns(pd.concat(rows, ignore_index=True)))


def read_json_table_export(path: Path, *, preferred: tuple[str, ...]) -> pd.DataFrame:
    """Read a flexible JSON row/column table export.

    ``preferred`` lists container keys to search before falling back to
    sequence mappings, row objects, or column maps.
    """

    return _read_json_table(path, preferred=preferred)


def _point_rows_to_candidates(
    points: pd.DataFrame,
    *,
    source: str,
    voxel_size_m: float,
    min_points: int,
    min_confidence: float,
) -> CandidateFrame:
    records: list[dict[str, object]] = []
    for (sequence_id, time_s), group in points.groupby(["sequence_id", "time_s"], sort=True):
        xyz = group[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
        for cluster_idx, members in enumerate(
            _voxel_connected_components(xyz, voxel_size_m=voxel_size_m, min_points=min_points)
        ):
            cluster = xyz[members]
            confidence = float(len(cluster))
            if confidence < float(min_confidence):
                continue
            centroid = cluster.mean(axis=0)
            spread = np.maximum(cluster.std(axis=0), 0.25)
            records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(time_s),
                    "source": source,
                    "track_id": f"{source}:{sequence_id}:{time_s}:{cluster_idx}",
                    "x_m": centroid[0],
                    "y_m": centroid[1],
                    "z_m": centroid[2],
                    "std_xy_m": float(max(spread[0], spread[1], 0.5)),
                    "std_z_m": float(max(spread[2], 0.5)),
                    "confidence": confidence,
                    "class_name": "uav",
                }
            )
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame.from_records(records)))


def _read_pcd(path: Path) -> pd.DataFrame:
    """Read a minimal PCD point cloud with ASCII or binary DATA sections."""

    raw = read_binary_export(path)
    marker = b"DATA"
    marker_index = raw.upper().find(marker)
    if marker_index < 0:
        raise ValueError(f"invalid PCD file without DATA header: {path}")
    line_end = raw.find(b"\n", marker_index)
    if line_end < 0:
        raise ValueError(f"invalid PCD file without data payload: {path}")
    header_text = raw[: line_end + 1].decode("utf-8", errors="ignore")
    payload = raw[line_end + 1 :]
    header = _parse_pcd_header(header_text)
    fields = header.get("fields", [])
    if not fields:
        raise ValueError(f"invalid PCD file without FIELDS: {path}")
    data_mode = str(header.get("data", "")).lower()
    if data_mode == "ascii":
        rows = []
        for line in payload.decode("utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < len(fields):
                continue
            rows.append({field: value for field, value in zip(fields, parts, strict=False)})
        return _normalize_point_frame(pd.DataFrame.from_records(rows), path=path)
    if data_mode == "binary":
        return _normalize_point_frame(_read_binary_pcd_payload(payload, header), path=path)
    if data_mode == "binary_compressed":
        return _normalize_point_frame(
            _read_binary_compressed_pcd_payload(payload, header),
            path=path,
        )
    raise ValueError(
        f"unsupported PCD DATA mode {data_mode!r}; expected ascii, binary, or binary_compressed"
    )


def _parse_pcd_header(header_text: str) -> dict[str, object]:
    header: dict[str, object] = {}
    for line in header_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        key = parts[0].lower()
        values = parts[1:]
        if key == "fields":
            header["fields"] = values
        elif key in {"size", "count"}:
            header[key] = [int(item) for item in values]
        elif key == "type":
            header[key] = values
        elif key in {"points", "width"} and values:
            header[key] = int(values[0])
        elif key == "data" and values:
            header[key] = values[0].lower()
    return header


def _read_binary_pcd_payload(payload: bytes, header: dict[str, object]) -> pd.DataFrame:
    fields = list(header.get("fields", []))
    sizes = list(header.get("size", [4] * len(fields)))
    types = list(header.get("type", ["F"] * len(fields)))
    counts = list(header.get("count", [1] * len(fields)))
    if len(sizes) != len(fields) or len(types) != len(fields):
        raise ValueError("PCD binary header has inconsistent FIELDS/SIZE/TYPE lengths")
    if len(counts) != len(fields):
        counts = [1] * len(fields)
    dtype_fields = []
    for field, size, type_code, count in zip(fields, sizes, types, counts, strict=False):
        dtype = _pcd_numpy_dtype(size=int(size), type_code=str(type_code))
        if int(count) == 1:
            dtype_fields.append((field, dtype))
        else:
            dtype_fields.append((field, dtype, (int(count),)))
    dtype = np.dtype(dtype_fields)
    point_count = int(header.get("points", header.get("width", 0)) or 0)
    if point_count <= 0:
        point_count = len(payload) // dtype.itemsize
    arr = np.frombuffer(payload[: point_count * dtype.itemsize], dtype=dtype, count=point_count)
    data: dict[str, np.ndarray] = {}
    for field in fields:
        values = arr[field]
        if getattr(values, "ndim", 1) > 1:
            values = values[:, 0]
        data[field] = values
    return pd.DataFrame(data)


def _read_binary_compressed_pcd_payload(payload: bytes, header: dict[str, object]) -> pd.DataFrame:
    import struct

    if len(payload) < 8:
        raise ValueError("PCD binary_compressed payload is missing size headers")
    compressed_size, uncompressed_size = struct.unpack_from("<II", payload, 0)
    compressed_start = 8
    compressed_end = compressed_start + int(compressed_size)
    if compressed_end > len(payload):
        raise ValueError("PCD binary_compressed payload is truncated")
    decompressed = _lzf_decompress(
        payload[compressed_start:compressed_end],
        expected_size=int(uncompressed_size),
    )
    fields = list(header.get("fields", []))
    sizes = list(header.get("size", [4] * len(fields)))
    types = list(header.get("type", ["F"] * len(fields)))
    counts = list(header.get("count", [1] * len(fields)))
    if len(sizes) != len(fields) or len(types) != len(fields):
        raise ValueError("PCD binary_compressed header has inconsistent FIELDS/SIZE/TYPE lengths")
    if len(counts) != len(fields):
        counts = [1] * len(fields)
    point_count = int(header.get("points", header.get("width", 0)) or 0)
    if point_count <= 0:
        raise ValueError("PCD binary_compressed header must include POINTS or WIDTH")
    data: dict[str, np.ndarray] = {}
    cursor = 0
    for field, size, type_code, count in zip(fields, sizes, types, counts, strict=False):
        dtype = np.dtype(_pcd_numpy_dtype(size=int(size), type_code=str(type_code)))
        component_count = int(count)
        byte_count = point_count * component_count * dtype.itemsize
        segment = decompressed[cursor : cursor + byte_count]
        if len(segment) != byte_count:
            raise ValueError("PCD binary_compressed decompressed payload is incomplete")
        values = np.frombuffer(segment, dtype=dtype, count=point_count * component_count)
        if component_count > 1:
            values = values.reshape(point_count, component_count)[:, 0]
        data[str(field)] = values
        cursor += byte_count
    return pd.DataFrame(data)


def _lzf_decompress(payload: bytes, *, expected_size: int) -> bytes:
    out = bytearray()
    idx = 0
    while idx < len(payload):
        ctrl = payload[idx]
        idx += 1
        if ctrl < 32:
            length = ctrl + 1
            chunk = payload[idx : idx + length]
            if len(chunk) != length:
                raise ValueError("truncated LZF literal run in PCD payload")
            out.extend(chunk)
            idx += length
            continue
        length = ctrl >> 5
        reference_offset = (ctrl & 0x1F) << 8
        if length == 7:
            if idx >= len(payload):
                raise ValueError("truncated LZF back-reference length in PCD payload")
            length += payload[idx]
            idx += 1
        if idx >= len(payload):
            raise ValueError("truncated LZF back-reference offset in PCD payload")
        reference_offset += payload[idx]
        idx += 1
        reference_index = len(out) - reference_offset - 1
        if reference_index < 0:
            raise ValueError("invalid LZF back-reference in PCD payload")
        for _ in range(length + 2):
            out.append(out[reference_index])
            reference_index += 1
    if len(out) != expected_size:
        raise ValueError(
            f"PCD binary_compressed payload expanded to {len(out)} bytes, expected {expected_size}"
        )
    return bytes(out)


def _pcd_numpy_dtype(*, size: int, type_code: str) -> str:
    code = type_code.upper()
    if code == "F":
        return "<f4" if size == 4 else "<f8"
    if code == "I":
        return {1: "<i1", 2: "<i2", 4: "<i4", 8: "<i8"}.get(size, "<i4")
    if code == "U":
        return {1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}.get(size, "<u4")
    raise ValueError(f"unsupported PCD type code: {type_code!r}")


def _read_numpy_point_cloud(path: Path) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.lib.npyio.NpzFile):
        key = "points" if "points" in payload.files else payload.files[0]
        arr = payload[key]
    else:
        arr = payload
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"NumPy point cloud must be shape (N, >=3), got {arr.shape}")
    frame = pd.DataFrame({"x_m": arr[:, 0], "y_m": arr[:, 1], "z_m": arr[:, 2]})
    if arr.shape[1] >= 4:
        frame["time_s"] = arr[:, 3]
    return _normalize_point_frame(frame, path=path)


def _read_binary_point_cloud(path: Path) -> pd.DataFrame:
    """Read a simple little-endian float32 point cloud.

    Common exported LiDAR ``.bin`` files store rows as ``x,y,z`` or
    ``x,y,z,intensity``.  The fourth channel is ignored here because sequence
    timestamps are frame-level metadata inferred from the filename unless an
    explicit ``time_s`` is supplied by the caller.
    """

    raw = np.frombuffer(read_binary_export(path), dtype="<f4")
    if raw.size < 3:
        raise ValueError(f"binary point cloud {path} contains fewer than 3 float32 values")
    if raw.size % 4 == 0:
        arr = raw.reshape(-1, 4)
    elif raw.size % 3 == 0:
        arr = raw.reshape(-1, 3)
    else:
        raise ValueError(
            f"binary point cloud {path} must contain float32 x,y,z or x,y,z,intensity rows"
        )
    frame = pd.DataFrame({"x_m": arr[:, 0], "y_m": arr[:, 1], "z_m": arr[:, 2]})
    return _normalize_point_frame(frame, path=path)


def _read_ply(path: Path) -> pd.DataFrame:
    header_lines, payload = _split_ply_header(read_binary_export(path), path=path)
    if not header_lines or header_lines[0].strip() != "ply":
        raise ValueError(f"invalid PLY file: {path}")
    data_format = "ascii"
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines[1:]:
        stripped = line.strip()
        if stripped.startswith("format"):
            parts = stripped.split()
            if len(parts) >= 2:
                data_format = parts[1].lower()
        if stripped.startswith("element vertex"):
            vertex_count = int(stripped.split()[-1])
            in_vertex = True
            continue
        if stripped.startswith("element") and not stripped.startswith("element vertex"):
            in_vertex = False
        if in_vertex and stripped.startswith("property "):
            parts = stripped.split()
            if len(parts) >= 2 and parts[1].lower() == "list":
                raise ValueError("PLY vertex list properties are not supported")
            if len(parts) >= 3:
                properties.append((parts[2], parts[1].lower()))
    if vertex_count <= 0 or not properties:
        raise ValueError(f"invalid PLY file: {path}")
    if data_format == "ascii":
        return _read_ascii_ply_payload(
            payload,
            properties=properties,
            vertex_count=vertex_count,
            path=path,
        )
    if data_format == "binary_little_endian":
        return _read_binary_ply_payload(
            payload,
            properties=properties,
            vertex_count=vertex_count,
            endian="<",
            path=path,
        )
    if data_format == "binary_big_endian":
        return _read_binary_ply_payload(
            payload,
            properties=properties,
            vertex_count=vertex_count,
            endian=">",
            path=path,
        )
    raise ValueError(f"unsupported PLY format {data_format!r}; expected ascii or binary")


def _split_ply_header(raw: bytes, *, path: Path) -> tuple[list[str], bytes]:
    offset = 0
    lines: list[str] = []
    for line in raw.splitlines(keepends=True):
        offset += len(line)
        text = line.decode("ascii", errors="ignore").strip()
        lines.append(text)
        if text == "end_header":
            return lines, raw[offset:]
    raise ValueError(f"invalid PLY file without end_header: {path}")


def _read_ascii_ply_payload(
    payload: bytes,
    *,
    properties: list[tuple[str, str]],
    vertex_count: int,
    path: Path,
) -> pd.DataFrame:
    rows = []
    property_names = [name for name, _type_name in properties]
    for line in payload.decode("utf-8", errors="ignore").splitlines()[:vertex_count]:
        parts = line.split()
        if len(parts) < len(property_names):
            continue
        rows.append({field: value for field, value in zip(property_names, parts, strict=False)})
    return _normalize_point_frame(pd.DataFrame.from_records(rows), path=path)


def _read_binary_ply_payload(
    payload: bytes,
    *,
    properties: list[tuple[str, str]],
    vertex_count: int,
    endian: str,
    path: Path,
) -> pd.DataFrame:
    dtype_fields = [
        (name, _ply_numpy_dtype(type_name=type_name, endian=endian))
        for name, type_name in properties
    ]
    dtype = np.dtype(dtype_fields)
    expected_bytes = int(vertex_count) * dtype.itemsize
    if len(payload) < expected_bytes:
        raise ValueError(f"binary PLY file has incomplete vertex payload: {path}")
    arr = np.frombuffer(payload[:expected_bytes], dtype=dtype, count=int(vertex_count))
    return _normalize_point_frame(
        pd.DataFrame({name: arr[name] for name, _type_name in properties}),
        path=path,
    )


def _ply_numpy_dtype(*, type_name: str, endian: str) -> str:
    normalized = type_name.lower()
    aliases = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    dtype = aliases.get(normalized)
    if dtype is None:
        raise ValueError(f"unsupported PLY property type: {type_name!r}")
    if dtype.endswith("1"):
        return dtype
    return f"{endian}{dtype}"


def _read_las(path: Path) -> pd.DataFrame:
    """Read uncompressed LAS point records as an exported point cloud."""

    raw = read_binary_export(path)
    if len(raw) < 227 or raw[:4] != b"LASF":
        raise ValueError(f"invalid LAS file: {path}")
    point_format_byte = raw[104]
    if point_format_byte & 0x80:
        return _read_las_with_laspy(path)
    point_format = point_format_byte & 0x3F
    if point_format > 10:
        raise ValueError(f"unsupported LAS point format: {point_format}")
    point_data_offset = _read_le_uint(raw, 96, 4, path=path)
    point_record_length = _read_le_uint(raw, 105, 2, path=path)
    point_count = _las_point_count(raw, path=path)
    if point_record_length < 12:
        raise ValueError(f"LAS point record length is too short: {point_record_length}")
    expected_end = int(point_data_offset) + int(point_count) * int(point_record_length)
    if expected_end > len(raw):
        raise ValueError(f"LAS point data is truncated: {path}")
    scale = np.array(_read_las_vector(raw, 131, path=path), dtype=float)
    offset = np.array(_read_las_vector(raw, 155, path=path), dtype=float)
    raw_xyz = np.empty((int(point_count), 3), dtype=np.int32)
    for column, column_offset in enumerate((0, 4, 8)):
        raw_xyz[:, column] = np.ndarray(
            shape=(int(point_count),),
            dtype="<i4",
            buffer=raw,
            offset=int(point_data_offset) + column_offset,
            strides=(int(point_record_length),),
        )
    xyz = raw_xyz.astype(float) * scale + offset
    return _normalize_point_frame(
        pd.DataFrame({"x_m": xyz[:, 0], "y_m": xyz[:, 1], "z_m": xyz[:, 2]}),
        path=path,
    )


def _read_las_with_laspy(path: Path) -> pd.DataFrame:
    """Read LAZ/LASzip point records through optional laspy support."""

    try:
        import laspy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "compressed LAZ/LASzip point clouds require the optional "
            "'laspy[lazrs]' dependency"
        ) from exc
    try:
        cloud = laspy.read(path)
    except Exception as exc:  # pragma: no cover - backend-specific diagnostics
        raise ValueError(f"failed to read LAZ/LASzip point cloud {path}: {exc}") from exc
    return _normalize_point_frame(
        pd.DataFrame(
            {
                "x_m": np.asarray(cloud.x, dtype=float),
                "y_m": np.asarray(cloud.y, dtype=float),
                "z_m": np.asarray(cloud.z, dtype=float),
            }
        ),
        path=path,
    )


def _las_point_count(raw: bytes, *, path: Path) -> int:
    legacy_count = _read_le_uint(raw, 107, 4, path=path)
    if legacy_count > 0:
        return int(legacy_count)
    if len(raw) >= 255:
        return int(_read_le_uint(raw, 247, 8, path=path))
    return 0


def _read_las_vector(raw: bytes, offset: int, *, path: Path) -> tuple[float, float, float]:
    import struct

    if offset + 24 > len(raw):
        raise ValueError(f"LAS header is truncated: {path}")
    return struct.unpack_from("<ddd", raw, offset)


def _read_le_uint(raw: bytes, offset: int, length: int, *, path: Path) -> int:
    if offset + length > len(raw):
        raise ValueError(f"LAS header is truncated: {path}")
    return int.from_bytes(raw[offset : offset + length], byteorder="little", signed=False)


def _normalize_point_frame(frame: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    aliases = {
        "x": "x_m",
        "y": "y_m",
        "z": "z_m",
        "X": "x_m",
        "Y": "y_m",
        "Z": "z_m",
    }
    out = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    missing = {"x_m", "y_m", "z_m"}.difference(out.columns)
    if missing:
        raise ValueError(f"point cloud {path} missing coordinate columns: {sorted(missing)}")
    for col in ("x_m", "y_m", "z_m"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.loc[np.isfinite(out[["x_m", "y_m", "z_m"]]).all(axis=1)].copy()


def _read_delimited_table(path: Path) -> pd.DataFrame:
    if data_file_suffix(path) == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")


def _read_json_table(path: Path, *, preferred: tuple[str, ...]) -> pd.DataFrame:
    payload = read_json_export_payload(path)
    records = _json_records_from_payload(payload, preferred=preferred)
    return _json_records_to_frame(records, path=path)


def read_json_export_payload(path: Path) -> Any:
    """Read a JSON table export payload, including newline-delimited JSON rows."""

    path = Path(path)
    if data_file_suffix(path) in {".jsonl", ".ndjson"}:
        rows = []
        for line in read_text_export(path).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
        return rows
    return json.loads(read_text_export(path))


def _json_records_from_payload(payload: Any, *, preferred: tuple[str, ...]) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    for key in preferred:
        nested = _lookup_case_insensitive(payload, key)
        if nested is not None:
            return _json_records_from_nested_container(
                payload,
                nested,
                preferred=preferred,
            )

    sequences = _lookup_case_insensitive(payload, "sequences")
    if sequences is not None:
        rows = _json_records_from_sequence_mapping(sequences, preferred=preferred)
        if rows:
            return rows

    if _looks_like_column_mapping(payload) or _looks_like_row(payload):
        return payload

    rows = _json_records_from_sequence_mapping(payload, preferred=preferred)
    return rows if rows else []


def _json_records_from_sequence_mapping(
    mapping: Any,
    *,
    preferred: tuple[str, ...],
) -> list[dict[str, Any]]:
    if isinstance(mapping, list):
        return _json_records_to_frame(mapping).to_dict("records")
    if not isinstance(mapping, dict):
        return []

    rows: list[dict[str, Any]] = []
    for sequence_id, value in mapping.items():
        if str(sequence_id).lower() in _JSON_METADATA_KEYS:
            continue
        records = _json_records_from_payload(value, preferred=preferred)
        try:
            frame = _json_records_to_frame(records)
        except ValueError:
            continue
        if frame.empty:
            continue
        if "sequence_id" not in frame.columns:
            frame["sequence_id"] = str(sequence_id)
        rows.extend(frame.to_dict("records"))
    return rows


def _json_records_from_nested_container(
    parent: dict[Any, Any],
    nested: Any,
    *,
    preferred: tuple[str, ...],
) -> Any:
    records = _json_records_from_payload(nested, preferred=preferred)
    defaults = _json_parent_row_defaults(parent)
    if not defaults or not isinstance(records, list):
        return records
    merged: list[Any] = []
    for record in records:
        if not isinstance(record, dict):
            merged.append(record)
            continue
        row = dict(defaults)
        row.update(record)
        merged.append(row)
    return merged


def _json_parent_row_defaults(parent: dict[Any, Any]) -> dict[Any, Any]:
    defaults: dict[Any, Any] = {}
    for key in (
        "sequence_id",
        "sequence",
        "seq",
        "scene",
        "scene_id",
        "source",
        "sensor",
        "modality",
        "header",
        "stamp",
        "time_s",
        "timestamp",
        "timestamp_s",
        "timestamp_ns",
        "timestamp_us",
        "timestamp_ms",
        "sec",
        "secs",
        "nanosec",
        "nsec",
        "nsecs",
        "frame_id",
        "child_frame_id",
    ):
        value = _lookup_case_insensitive(parent, key)
        if value is not None:
            defaults[key] = value
    return defaults


def _json_records_to_frame(records: Any, *, path: Path | None = None) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, dict):
        if _looks_like_column_mapping(records):
            return pd.DataFrame(records)
        if _looks_like_row(records):
            return pd.DataFrame.from_records([_flatten_tracking_record(records)])
    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in records):
            return pd.DataFrame.from_records(
                [_flatten_tracking_record(item) for item in records]
            )
    label = str(path) if path is not None else "JSON payload"
    raise ValueError(f"JSON table {label} does not contain row objects")


def _lookup_case_insensitive(mapping: dict[Any, Any], key: str) -> Any | None:
    for candidate, value in mapping.items():
        if str(candidate).lower() == key.lower():
            return value
    return None


_JSON_ROW_HINT_KEYS = {
    "time_s",
    "timestamp",
    "timestamp_s",
    "timestamp_ns",
    "timestamp_us",
    "timestamp_ms",
    "stamp",
    "sec",
    "secs",
    "nanosec",
    "nsec",
    "nsecs",
    "x_m",
    "x",
    "y_m",
    "y",
    "z_m",
    "z",
    "header",
    "frame_id",
    "child_frame_id",
    "pose",
    "position",
    "point",
    "transform",
    "translation",
    "center",
    "location",
    "coordinates",
    "xyz",
}
_JSON_METADATA_KEYS = {
    "schema",
    "version",
    "metadata",
    "meta",
    "description",
    "exports",
    "calibration",
    "sensors",
    "classes",
    "class_map",
}


def _looks_like_row(payload: dict[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    return bool(keys.intersection(_JSON_ROW_HINT_KEYS))


def _looks_like_column_mapping(payload: dict[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    if not keys.intersection(_JSON_ROW_HINT_KEYS):
        return False
    column_values = [
        value
        for key, value in payload.items()
        if str(key).lower() in _JSON_ROW_HINT_KEYS and isinstance(value, (list, tuple))
    ]
    return bool(column_values)


def _flatten_tracking_record(record: dict[Any, Any]) -> dict[Any, Any]:
    """Flatten common ROS-shaped JSON position rows into table columns."""

    out = dict(record)
    header = _lookup_case_insensitive(out, "header")
    if isinstance(header, dict):
        _copy_stamp_time(out, _lookup_case_insensitive(header, "stamp"))
        frame_id = _lookup_case_insensitive(header, "frame_id")
        _set_if_missing(out, "frame_id", frame_id)
        _set_if_missing(out, "source", frame_id, aliases=("sensor", "modality"))
    if not _has_time_key(out):
        _copy_stamp_time(out, _lookup_case_insensitive(out, "stamp"))
    frame_id = _lookup_case_insensitive(out, "frame_id")
    if frame_id is not None:
        _set_if_missing(out, "source", frame_id, aliases=("sensor", "modality"))
    child_frame_id = _lookup_case_insensitive(out, "child_frame_id")
    _set_if_missing(
        out,
        "track_id",
        child_frame_id,
        aliases=("track", "id", "object_id", "cluster_id", "instance_id"),
    )
    xyz = _xyz_from_nested_record(out)
    if xyz is not None:
        _set_if_missing(
            out,
            "x_m",
            xyz[0],
            aliases=("x", "east_m", "pos_x", "center_x", "cx", "px"),
        )
        _set_if_missing(
            out,
            "y_m",
            xyz[1],
            aliases=("y", "north_m", "pos_y", "center_y", "cy", "py"),
        )
        _set_if_missing(
            out,
            "z_m",
            xyz[2],
            aliases=("z", "up_m", "pos_z", "center_z", "cz", "pz"),
        )
    return out


def _copy_stamp_time(out: dict[Any, Any], stamp: Any) -> None:
    if _has_time_key(out):
        return
    time_s = _stamp_to_seconds(stamp)
    if time_s is not None:
        out["time_s"] = time_s


def _stamp_to_seconds(stamp: Any) -> float | None:
    if isinstance(stamp, dict):
        nested = _lookup_case_insensitive(stamp, "stamp")
        if nested is not None:
            nested_time = _stamp_to_seconds(nested)
            if nested_time is not None:
                return nested_time
        seconds = _first_mapping_value(stamp, ("sec", "secs", "seconds"))
        nanoseconds = _first_mapping_value(
            stamp,
            ("nanosec", "nsec", "nsecs", "nanoseconds"),
        )
        if seconds is not None:
            try:
                return float(seconds) + (float(nanoseconds or 0.0) * 1.0e-9)
            except (TypeError, ValueError):
                return None
        numeric = _first_mapping_value(
            stamp,
            ("time_s", "timestamp_s", "timestamp", "stamp", "time"),
        )
        return _float_or_none(numeric)
    return _float_or_none(stamp)


def _xyz_from_nested_record(value: Any, *, depth: int = 0) -> tuple[float, float, float] | None:
    if depth > 8:
        return None
    if isinstance(value, dict):
        xyz = _xyz_from_mapping(value)
        if xyz is not None:
            return xyz
        for key in (
            "position",
            "point",
            "translation",
            "center",
            "location",
            "coordinates",
            "xyz",
            "pose",
            "transform",
            "state",
            "measurement",
        ):
            nested = _lookup_case_insensitive(value, key)
            if nested is None:
                continue
            xyz = _xyz_from_nested_record(nested, depth=depth + 1)
            if xyz is not None:
                return xyz
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        numbers = _numeric_sequence(value)
        if len(numbers) >= 3:
            return (numbers[0], numbers[1], numbers[2])
    return None


def _xyz_from_mapping(mapping: dict[Any, Any]) -> tuple[float, float, float] | None:
    x = _first_float_mapping_value(
        mapping,
        ("x_m", "x", "east_m", "pos_x", "center_x", "cx", "px"),
    )
    y = _first_float_mapping_value(
        mapping,
        ("y_m", "y", "north_m", "pos_y", "center_y", "cy", "py"),
    )
    z = _first_float_mapping_value(
        mapping,
        ("z_m", "z", "up_m", "pos_z", "center_z", "cz", "pz"),
    )
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def _first_float_mapping_value(mapping: dict[Any, Any], keys: Iterable[str]) -> float | None:
    return _float_or_none(_first_mapping_value(mapping, keys))


def _first_mapping_value(mapping: dict[Any, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        value = _lookup_case_insensitive(mapping, key)
        if value is not None:
            return value
    return None


def _set_if_missing(
    mapping: dict[Any, Any],
    key: str,
    value: Any,
    *,
    aliases: Iterable[str] = (),
) -> None:
    if value is None:
        return
    if not _has_any_key(mapping, (key, *tuple(aliases))):
        mapping[key] = value


def _has_time_key(mapping: dict[Any, Any]) -> bool:
    return _has_any_key(
        mapping,
        (
            "time_s",
            "timestamp",
            "timestamp_s",
            "timestamp_ns",
            "timestamp_us",
            "timestamp_ms",
            "sec",
            "secs",
            "nanosec",
            "nsec",
            "nsecs",
        ),
    )


def _has_any_key(mapping: dict[Any, Any], keys: Iterable[str]) -> bool:
    present = {str(key).lower() for key in mapping}
    return any(key.lower() in present for key in keys)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return _numeric_sequence(json.loads(text))
        except json.JSONDecodeError:
            trimmed = text.strip("[]()")
            delimiter = "," if "," in trimmed else None
            return _numeric_sequence(trimmed.split(delimiter))
    if isinstance(value, dict):
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        numbers: list[float] = []
        for item in value:
            try:
                numbers.append(float(item))
            except (TypeError, ValueError):
                return []
        return numbers
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        return []
    return []


def _read_numpy_trajectory_table(path: Path) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.lib.npyio.NpzFile):
        key = _first_npz_key(
            payload,
            preferred=("trajectory", "trajectories", "truth", "candidates", "detections", "poses", "data"),
        )
        arr = payload[key]
    else:
        arr = payload
    arr = np.asarray(arr)
    if arr.dtype.names:
        return pd.DataFrame.from_records(arr)
    if arr.ndim == 1 and arr.shape[0] >= 3:
        columns = ["x_m", "y_m", "z_m"]
        frame = pd.DataFrame([arr[:3]], columns=columns)
        frame.insert(0, "time_s", infer_time_s_from_filename(path))
        if arr.shape[0] >= 4:
            frame["confidence"] = arr[3]
        return frame
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"NumPy trajectory table must be shape (N, >=4), got {arr.shape}")
    columns = ["time_s", "x_m", "y_m", "z_m"]
    if arr.shape[1] >= 5:
        columns.append("confidence")
    frame = pd.DataFrame(arr[:, : len(columns)], columns=columns)
    return frame


def _first_npz_key(payload: np.lib.npyio.NpzFile, *, preferred: tuple[str, ...]) -> str:
    lower_to_key = {key.lower(): key for key in payload.files}
    for key in preferred:
        matched = lower_to_key.get(key.lower())
        if matched is not None:
            return matched
    return payload.files[0]


def _has_any_column(frame: pd.DataFrame, names: tuple[str, ...]) -> bool:
    lower = {str(column).lower() for column in frame.columns}
    return any(name.lower() in lower for name in names)


def _voxel_connected_components(
    xyz: np.ndarray,
    *,
    voxel_size_m: float,
    min_points: int,
) -> list[np.ndarray]:
    if xyz.size == 0:
        return []
    voxels = np.floor(xyz / float(voxel_size_m)).astype(int)
    voxel_to_points: dict[tuple[int, int, int], list[int]] = {}
    for idx, voxel in enumerate(voxels):
        voxel_to_points.setdefault(tuple(int(v) for v in voxel), []).append(idx)
    visited: set[tuple[int, int, int]] = set()
    components: list[np.ndarray] = []
    for voxel in voxel_to_points:
        if voxel in visited:
            continue
        queue: deque[tuple[int, int, int]] = deque([voxel])
        visited.add(voxel)
        point_indices: list[int] = []
        while queue:
            current = queue.popleft()
            point_indices.extend(voxel_to_points[current])
            for neighbor in _neighbors26(current):
                if neighbor in voxel_to_points and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(point_indices) >= int(min_points):
            components.append(np.asarray(point_indices, dtype=int))
    return components


def _neighbors26(voxel: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    x, y, z = voxel
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                yield (x + dx, y + dy, z + dz)
