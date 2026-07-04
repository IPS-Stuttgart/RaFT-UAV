#!/usr/bin/env python3
"""Search local dataset/code trees for Fortem radar-origin candidates.

This script is intentionally conservative: it does not need RaFT-UAV imports and
does not modify files. It scans text-like files for:
  - radar/Fortem/origin/conversion context terms,
  - latitude/longitude/altitude-like assignments,
  - numeric coordinate triples near an optional reference point,
  - MATLAB/geodetic conversion snippets.

Outputs:
  - radar_origin_search_matches.csv
  - radar_origin_coordinate_candidates.csv
  - radar_origin_search_report.json

Example:
  python tools/find_radar_origin_candidates.py \
    data/raw/AADM2025Dryad config matlab_scripts \
    --reference-lat 35.72750947 \
    --reference-lon -78.69595819 \
    --output-dir outputs/radar-origin-search
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable, Iterator

TEXT_EXTENSIONS = {
    ".m",
    ".mlx",
    ".txt",
    ".csv",
    ".json",
    ".geojson",
    ".kml",
    ".xml",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".matlab",
    ".dat",
    ".log",
}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
}

CONTEXT_TERMS = [
    "radar",
    "fortem",
    "range",
    "azimuth",
    "elevation",
    "geodetic2enu",
    "enu2geodetic",
    "lla",
    "lw1",
    "origin",
    "sensor",
    "tower",
    "mount",
    "antenna",
    "site",
    "latitude",
    "longitude",
    "altitude",
    "lat",
    "lon",
    "gps",
    "wgs84",
    "ellipsoid",
    "msl",
    "agl",
]
HIGH_VALUE_TERMS = [
    "radar",
    "fortem",
    "geodetic2enu",
    "enu2geodetic",
    "origin",
    "sensor",
    "mount",
    "antenna",
]

MATCH_FIELDS = ["score", "path", "line_number", "matched_terms", "line", "context"]
COORDINATE_FIELDS = [
    "score",
    "path",
    "line_number",
    "source",
    "name",
    "ordering",
    "latitude",
    "longitude",
    "altitude",
    "distance_to_reference_km",
    "near_reference",
    "line",
    "context",
]

# Assignment-like coordinates. Handles:
# radarLat = 35.7
# radar_origin_lla = [35.7 -78.6 113]
# "latitude": 35.7, "longitude": -78.6, "altitude": 113
ASSIGNMENT_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_\-. ]{0,80}?"
    r"(?:lat|latitude|lon|lng|longitude|alt|altitude|origin|lla|radar|fortem|sensor)"
    r"[A-Za-z0-9_\-. ]{0,80})"
    r"\s*(?:=|:)\s*"
    r"(?P<value>\[[^\]\n]{1,160}\]|\([^\)\n]{1,160}\)|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
TRIPLE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<a>[-+]?\d{1,3}\.\d{4,})"
    r"\s*[,;\s]\s*"
    r"(?P<b>[-+]?\d{1,3}\.\d{4,})"
    r"(?:\s*[,;\s]\s*(?P<c>[-+]?\d+(?:\.\d+)?))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Files or directories to scan")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radar-origin-search"))
    parser.add_argument("--reference-lat", type=float, default=None)
    parser.add_argument("--reference-lon", type=float, default=None)
    parser.add_argument("--max-distance-km", type=float, default=5.0)
    parser.add_argument("--max-file-size-mb", type=float, default=25.0)
    parser.add_argument("--include-extension", action="append", default=[])
    parser.add_argument("--context-radius", type=int, default=2, help="Lines before/after match")
    return parser.parse_args()


def iter_files(
    roots: Iterable[Path],
    *,
    max_file_size_mb: float,
    extra_exts: Iterable[str],
) -> Iterator[Path]:
    extensions = set(TEXT_EXTENSIONS)
    extensions.update(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extra_exts)
    max_bytes = int(max_file_size_mb * 1024 * 1024)
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            if _is_text_candidate(root, extensions, max_bytes):
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                path = Path(dirpath) / filename
                if _is_text_candidate(path, extensions, max_bytes):
                    yield path


def _is_text_candidate(path: Path, extensions: set[str], max_bytes: int) -> bool:
    try:
        if path.stat().st_size > max_bytes:
            return False
    except OSError:
        return False
    return path.suffix.lower() in extensions or path.name.lower() in {"readme", "metadata"}


def read_text(path: Path) -> str | None:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="replace")
        except Exception:
            continue
    return None


def term_score(text: str) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in CONTEXT_TERMS)


def high_value_score(text: str) -> int:
    lower = text.lower()
    return sum(3 * lower.count(term) for term in HIGH_VALUE_TERMS)


def snippet(lines: list[str], index: int, radius: int) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(f"{i + 1}: {lines[i].rstrip()}" for i in range(start, end))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def extract_numeric_values(raw: str) -> list[float]:
    return [float(match.group(0)) for match in NUMBER_RE.finditer(raw)]


def plausible_lat_lon(a: float, b: float) -> bool:
    return -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0


def coordinate_candidates_from_line(
    *,
    path: Path,
    line_number: int,
    line: str,
    context: str,
    reference_lat: float | None,
    reference_lon: float | None,
    max_distance_km: float,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    # Explicit triples/pairs anywhere in the line.
    for match in TRIPLE_RE.finditer(line):
        nums = [float(v) for v in (match.group("a"), match.group("b")) if v is not None]
        if match.group("c") is not None:
            nums.append(float(match.group("c")))
        candidates.extend(
            _make_coordinate_records(
                path=path,
                line_number=line_number,
                source="numeric-sequence",
                name=None,
                values=nums,
                line=line,
                context=context,
                reference_lat=reference_lat,
                reference_lon=reference_lon,
                max_distance_km=max_distance_km,
            )
        )

    # Assignment-like records.
    for match in ASSIGNMENT_RE.finditer(line):
        values = extract_numeric_values(match.group("value"))
        candidates.extend(
            _make_coordinate_records(
                path=path,
                line_number=line_number,
                source="assignment",
                name=match.group("name").strip(),
                values=values,
                line=line,
                context=context,
                reference_lat=reference_lat,
                reference_lon=reference_lon,
                max_distance_km=max_distance_km,
            )
        )
    return candidates


def _make_coordinate_records(
    *,
    path: Path,
    line_number: int,
    source: str,
    name: str | None,
    values: list[float],
    line: str,
    context: str,
    reference_lat: float | None,
    reference_lon: float | None,
    max_distance_km: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if len(values) < 2:
        return records

    # Try both LAT,LON and LON,LAT because KML commonly uses lon,lat,alt.
    orderings = [
        ("lat_lon", values[0], values[1], values[2] if len(values) >= 3 else None),
        ("lon_lat", values[1], values[0], values[2] if len(values) >= 3 else None),
    ]
    for ordering, lat, lon, alt in orderings:
        if not plausible_lat_lon(lat, lon):
            continue
        distance_km = None
        near_reference = None
        if reference_lat is not None and reference_lon is not None:
            distance_km = haversine_km(reference_lat, reference_lon, lat, lon)
            near_reference = distance_km <= max_distance_km

        local_context = f"{line}\n{context}"
        score = term_score(local_context) + high_value_score(local_context)
        if near_reference is True:
            score += 50
        if "radar" in local_context.lower() or "fortem" in local_context.lower():
            score += 25
        if name and any(term in name.lower() for term in ("radar", "fortem", "sensor", "origin", "lla")):
            score += 20

        records.append(
            {
                "score": score,
                "path": str(path),
                "line_number": line_number,
                "source": source,
                "name": name or "",
                "ordering": ordering,
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "distance_to_reference_km": distance_km,
                "near_reference": near_reference,
                "line": line.strip(),
                "context": context,
            }
        )
    return records


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matches: list[dict[str, object]] = []
    coord_candidates: list[dict[str, object]] = []
    files_scanned = 0

    for path in iter_files(
        args.roots,
        max_file_size_mb=args.max_file_size_mb,
        extra_exts=args.include_extension,
    ):
        text = read_text(path)
        if text is None:
            continue
        files_scanned += 1
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            lower = line.lower()
            matched_terms = [term for term in CONTEXT_TERMS if term in lower]
            if matched_terms:
                ctx = snippet(lines, idx, args.context_radius)
                matches.append(
                    {
                        "score": term_score(ctx) + high_value_score(ctx),
                        "path": str(path),
                        "line_number": idx + 1,
                        "matched_terms": ",".join(matched_terms),
                        "line": line.strip(),
                        "context": ctx,
                    }
                )

            if any(ch.isdigit() for ch in line):
                ctx = snippet(lines, idx, args.context_radius)
                coord_candidates.extend(
                    coordinate_candidates_from_line(
                        path=path,
                        line_number=idx + 1,
                        line=line,
                        context=ctx,
                        reference_lat=args.reference_lat,
                        reference_lon=args.reference_lon,
                        max_distance_km=args.max_distance_km,
                    )
                )

    matches.sort(key=lambda row: (-int(row["score"]), str(row["path"]), int(row["line_number"])))
    coord_candidates.sort(
        key=lambda row: (
            -int(row["score"]),
            float("inf")
            if row["distance_to_reference_km"] is None
            else float(row["distance_to_reference_km"]),
            str(row["path"]),
            int(row["line_number"]),
        )
    )

    matches_csv = args.output_dir / "radar_origin_search_matches.csv"
    candidates_csv = args.output_dir / "radar_origin_coordinate_candidates.csv"
    report_json = args.output_dir / "radar_origin_search_report.json"

    write_csv(matches_csv, matches, fieldnames=MATCH_FIELDS)
    write_csv(candidates_csv, coord_candidates, fieldnames=COORDINATE_FIELDS)

    report = {
        "roots": [str(path) for path in args.roots],
        "files_scanned": files_scanned,
        "matches_csv": str(matches_csv),
        "coordinate_candidates_csv": str(candidates_csv),
        "reference_lat": args.reference_lat,
        "reference_lon": args.reference_lon,
        "max_distance_km": args.max_distance_km,
        "top_coordinate_candidates": coord_candidates[:25],
        "top_context_matches": matches[:25],
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"files_scanned={files_scanned}")
    print(f"matches_csv={matches_csv}")
    print(f"coordinate_candidates_csv={candidates_csv}")
    print(f"report_json={report_json}")
    if coord_candidates:
        print("\nTop coordinate candidates:")
        for row in coord_candidates[:10]:
            dist = row["distance_to_reference_km"]
            dist_str = "n/a" if dist is None else f"{float(dist):.3f} km"
            print(
                f"score={row['score']:>3} dist={dist_str:>10} "
                f"lat={row['latitude']:.8f} lon={row['longitude']:.8f} "
                f"alt={row['altitude']} {row['path']}:{row['line_number']} "
                f"{row['name'] or row['source']}"
            )
    return 0


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    if fieldnames is None:
        fields: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        if not fields:
            fields = ["score", "path", "line_number", "line", "context"]
    else:
        fields = list(fieldnames)
        if rows:
            seen = set(fields)
            for row in rows:
                for key in row:
                    if key not in seen:
                        fields.append(key)
                        seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
