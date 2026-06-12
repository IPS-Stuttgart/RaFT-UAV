"""I/O helpers for normalized MMUAD tracking candidates."""

from __future__ import annotations

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
    """Load a normalized candidate table from CSV/TXT/JSON or NumPy trajectory rows."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_candidate_csv(
            path,
            default_sequence_id=default_sequence_id,
            source=source,
        )
    if suffix in {".tsv", ".txt"}:
        raw = _read_delimited_table(path)
    elif suffix == ".json":
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
    """Load a normalized truth table from CSV/TXT/JSON or NumPy trajectory rows."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_truth_csv(path, default_sequence_id=default_sequence_id)
    if suffix in {".tsv", ".txt"}:
        rows = normalize_truth_columns(
            _read_delimited_table(path),
            default_sequence_id=default_sequence_id,
        )
    elif suffix == ".json":
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

    CSV/TSV/TXT/JSON, NumPy, PCD, PLY, and simple float32 ``.bin`` files are
    supported as pragmatic exported-data bridges.  This is **not** a native
    Livox packet reader.  Files without per-point timestamps are treated as one
    frame; ``time_s`` is inferred from the filename when it contains a numeric
    token, otherwise it defaults to ``0.0``.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    source = source or path.stem.replace("_points", "-cluster")
    if suffix in {".csv", ".tsv", ".txt"}:
        points = _read_point_cloud_csv(path)
    elif suffix == ".json":
        points = _read_point_cloud_json(path)
    elif suffix in {".npy", ".npz"}:
        points = _read_numpy_point_cloud(path)
    elif suffix == ".pcd":
        points = _read_pcd(path)
    elif suffix == ".ply":
        points = _read_ascii_ply(path)
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

    raw = Path(path).read_bytes()
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
    raise ValueError(f"unsupported PCD DATA mode {data_mode!r}; expected ascii or binary")


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

    raw = np.fromfile(path, dtype="<f4")
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


def _read_ascii_ply(path: Path) -> pd.DataFrame:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"invalid PLY file: {path}")
    vertex_count = 0
    properties: list[str] = []
    data_start = None
    in_vertex = False
    for idx, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped.startswith("format") and "ascii" not in stripped:
            raise ValueError("only ASCII PLY files are supported")
        if stripped.startswith("element vertex"):
            vertex_count = int(stripped.split()[-1])
            in_vertex = True
            continue
        if stripped.startswith("element") and not stripped.startswith("element vertex"):
            in_vertex = False
        if in_vertex and stripped.startswith("property"):
            properties.append(stripped.split()[-1])
        if stripped == "end_header":
            data_start = idx + 1
            break
    if data_start is None or vertex_count <= 0 or not properties:
        raise ValueError(f"invalid ASCII PLY file: {path}")
    rows = []
    for line in lines[data_start : data_start + vertex_count]:
        parts = line.split()
        if len(parts) < len(properties):
            continue
        rows.append({field: value for field, value in zip(properties, parts, strict=False)})
    return _normalize_point_frame(pd.DataFrame.from_records(rows), path=path)


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
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")


def _read_json_table(path: Path, *, preferred: tuple[str, ...]) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = _json_records_from_payload(payload, preferred=preferred)
    return _json_records_to_frame(records, path=path)


def _json_records_from_payload(payload: Any, *, preferred: tuple[str, ...]) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    for key in preferred:
        nested = _lookup_case_insensitive(payload, key)
        if nested is not None:
            return _json_records_from_payload(nested, preferred=preferred)

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


def _json_records_to_frame(records: Any, *, path: Path | None = None) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, dict):
        if _looks_like_column_mapping(records):
            return pd.DataFrame(records)
        if _looks_like_row(records):
            return pd.DataFrame.from_records([records])
    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in records):
            return pd.DataFrame.from_records(records)
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
    "x_m",
    "x",
    "y_m",
    "y",
    "z_m",
    "z",
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
