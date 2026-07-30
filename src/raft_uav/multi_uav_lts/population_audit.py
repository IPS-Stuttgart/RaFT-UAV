"""Audit whether Multi-UAV LTS identities are fully seeded on frame one."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ._records import parse_detection_text


@dataclass(frozen=True)
class SequencePopulationAudit:
    sequence: str
    identity_count: int
    frame_one_identity_count: int
    late_birth_identity_count: int
    late_birth_ids: tuple[int, ...]
    reappearing_identity_count: int
    reappearing_ids: tuple[int, ...]
    maximum_gap_frames: int


@dataclass(frozen=True)
class PopulationAudit:
    truth_dir: str
    sequence_count: int
    identity_count: int
    frame_one_identity_count: int
    late_birth_identity_count: int
    late_birth_fraction: float
    sequences_with_late_births: int
    reappearing_identity_count: int
    maximum_gap_frames: int
    sequences: tuple[SequencePopulationAudit, ...]


def _truth_paths(truth_dir: Path) -> list[Path]:
    if not truth_dir.exists():
        raise FileNotFoundError(f"truth directory does not exist: {truth_dir}")
    if not truth_dir.is_dir():
        raise NotADirectoryError(f"truth path is not a directory: {truth_dir}")
    truth_paths = sorted(truth_dir.glob("*.txt"))
    if not truth_paths:
        raise ValueError(f"truth directory contains no .txt files: {truth_dir}")
    return truth_paths


def _paths_alias(left: Path, right: Path) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    try:
        return left_path.samefile(right_path)
    except OSError:
        return left_path.resolve() == right_path.resolve()


def _reject_population_audit_output_aliases(
    audit: PopulationAudit,
    json_path: Path,
    csv_path: Path,
) -> None:
    json_output = Path(json_path)
    csv_output = Path(csv_path)
    if _paths_alias(json_output, csv_output):
        raise ValueError("population audit JSON and CSV outputs must differ")

    truth_dir = Path(audit.truth_dir)
    if not truth_dir.is_dir():
        return
    truth_paths = tuple(truth_dir.glob("*.txt"))
    for label, output_path in (
        ("JSON output", json_output),
        ("CSV output", csv_output),
    ):
        for truth_path in truth_paths:
            if _paths_alias(output_path, truth_path):
                raise ValueError(
                    f"population audit {label} must differ from truth input file: "
                    f"{truth_path}"
                )


def audit_first_frame_population(truth_dir: Path) -> PopulationAudit:
    sequence_rows: list[SequencePopulationAudit] = []
    for truth_path in _truth_paths(truth_dir):
        rows = parse_detection_text(
            truth_path.read_text(encoding="utf-8"), source=str(truth_path)
        )
        by_id: dict[int, list[int]] = {}
        for row in rows:
            by_id.setdefault(row.object_id, []).append(row.frame_id)
        late_births: list[int] = []
        reappearing: list[int] = []
        maximum_gap = 0
        frame_one_count = 0
        for object_id, frames in sorted(by_id.items()):
            unique_frames = sorted(set(frames))
            if unique_frames[0] == 1:
                frame_one_count += 1
            else:
                late_births.append(object_id)
            gaps = [
                right - left - 1
                for left, right in zip(unique_frames, unique_frames[1:])
            ]
            identity_maximum_gap = max(gaps, default=0)
            maximum_gap = max(maximum_gap, identity_maximum_gap)
            if identity_maximum_gap > 0:
                reappearing.append(object_id)
        sequence_rows.append(
            SequencePopulationAudit(
                sequence=truth_path.stem,
                identity_count=len(by_id),
                frame_one_identity_count=frame_one_count,
                late_birth_identity_count=len(late_births),
                late_birth_ids=tuple(late_births),
                reappearing_identity_count=len(reappearing),
                reappearing_ids=tuple(reappearing),
                maximum_gap_frames=maximum_gap,
            )
        )

    identity_count = sum(row.identity_count for row in sequence_rows)
    late_birth_count = sum(row.late_birth_identity_count for row in sequence_rows)
    return PopulationAudit(
        truth_dir=str(truth_dir),
        sequence_count=len(sequence_rows),
        identity_count=identity_count,
        frame_one_identity_count=sum(
            row.frame_one_identity_count for row in sequence_rows
        ),
        late_birth_identity_count=late_birth_count,
        late_birth_fraction=late_birth_count / max(1, identity_count),
        sequences_with_late_births=sum(
            row.late_birth_identity_count > 0 for row in sequence_rows
        ),
        reappearing_identity_count=sum(
            row.reappearing_identity_count for row in sequence_rows
        ),
        maximum_gap_frames=max(
            (row.maximum_gap_frames for row in sequence_rows), default=0
        ),
        sequences=tuple(sequence_rows),
    )


def write_population_audit(
    audit: PopulationAudit, json_path: Path, csv_path: Path
) -> None:
    _reject_population_audit_output_aliases(audit, json_path, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(audit), indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = list(SequencePopulationAudit.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in audit.sequences:
            payload = asdict(row)
            payload["late_birth_ids"] = " ".join(map(str, row.late_birth_ids))
            payload["reappearing_ids"] = " ".join(map(str, row.reappearing_ids))
            writer.writerow(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth_dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sequence-summary-csv", type=Path, required=True)
    parser.add_argument(
        "--require-no-late-births",
        action="store_true",
        help="return nonzero when any identity first appears after frame one",
    )
    args = parser.parse_args(argv)
    audit = audit_first_frame_population(args.truth_dir)
    write_population_audit(audit, args.output_json, args.sequence_summary_csv)
    print(f"population_identity_count={audit.identity_count}")
    print(f"population_late_birth_identity_count={audit.late_birth_identity_count}")
    print(f"population_late_birth_fraction={audit.late_birth_fraction:.6f}")
    if args.require_no_late_births and audit.late_birth_identity_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
