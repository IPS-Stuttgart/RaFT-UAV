"""Fuse complementary Multi-UAV LTS proposal banks without premature NMS."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ._records import Detection, format_detection, parse_detection_text, prediction_texts

_SUMMARY_SCHEMA = "raft-uav-multi-uav-lts-proposal-bank-fusion-v1"


@dataclass(frozen=True)
class ProposalSourceSummary:
    name: str
    path: str
    sequence_count: int
    row_count: int


@dataclass(frozen=True)
class FusedSequenceSummary:
    sequence: str
    row_count: int
    source_row_counts: Mapping[str, int]


@dataclass(frozen=True)
class ProposalBankFusionSummary:
    schema: str
    output_dir: str
    source_count: int
    sequence_count: int
    row_count: int
    sources: tuple[ProposalSourceSummary, ...]
    sequences: tuple[FusedSequenceSummary, ...]


def fuse_proposal_banks(
    sources: Mapping[str, Path],
    output_dir: Path,
    *,
    sequences: Sequence[str] | None = None,
) -> ProposalBankFusionSummary:
    """Union proposal sources while keeping every source observation available."""

    if not sources:
        raise ValueError("at least one proposal source is required")
    normalized: dict[str, Path] = {}
    for raw_name, raw_path in sources.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("proposal source names must be non-empty")
        if name in normalized:
            raise ValueError(f"duplicate proposal source name: {name}")
        path = Path(raw_path)
        _validate_source_output_separation(path, output_dir)
        normalized[name] = path

    source_texts = {name: prediction_texts(path) for name, path in normalized.items()}
    available = sorted(
        {Path(file_name).stem for texts in source_texts.values() for file_name in texts}
    )
    requested = tuple(dict.fromkeys(str(value) for value in (sequences or ())))
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"requested sequences absent from every source: {', '.join(missing)}")
    selected = requested or tuple(available)

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.txt"):
        stale.unlink()

    source_counts = {name: 0 for name in normalized}
    source_sequence_counts = {name: 0 for name in normalized}
    sequence_summaries: list[FusedSequenceSummary] = []
    total_rows = 0
    for sequence in selected:
        fused: list[Detection] = []
        per_source: dict[str, int] = {}
        next_id_by_frame: dict[int, int] = {}
        for name in sorted(normalized):
            text = source_texts[name].get(f"{sequence}.txt")
            if text is None:
                per_source[name] = 0
                continue
            rows = parse_detection_text(
                text,
                source=f"{normalized[name]}:{sequence}.txt",
            )
            per_source[name] = len(rows)
            source_counts[name] += len(rows)
            source_sequence_counts[name] += 1
            for row in sorted(
                rows,
                key=lambda value: (
                    value.frame_id,
                    value.object_id,
                    value.x1,
                    value.y1,
                    value.width,
                    value.height,
                    -value.confidence,
                ),
            ):
                next_id = next_id_by_frame.get(row.frame_id, 1)
                fused.append(
                    Detection(
                        row.frame_id,
                        next_id,
                        row.x1,
                        row.y1,
                        row.width,
                        row.height,
                        row.confidence,
                        row.class_id,
                        row.visibility,
                    )
                )
                next_id_by_frame[row.frame_id] = next_id + 1
        fused.sort(key=lambda row: (row.frame_id, row.object_id))
        (output_dir / f"{sequence}.txt").write_text(
            "".join(format_detection(row) + "\n" for row in fused),
            encoding="utf-8",
        )
        total_rows += len(fused)
        sequence_summaries.append(
            FusedSequenceSummary(
                sequence=sequence,
                row_count=len(fused),
                source_row_counts=per_source,
            )
        )

    source_summaries = tuple(
        ProposalSourceSummary(
            name=name,
            path=str(normalized[name]),
            sequence_count=source_sequence_counts[name],
            row_count=source_counts[name],
        )
        for name in sorted(normalized)
    )
    return ProposalBankFusionSummary(
        schema=_SUMMARY_SCHEMA,
        output_dir=str(output_dir),
        source_count=len(source_summaries),
        sequence_count=len(sequence_summaries),
        row_count=total_rows,
        sources=source_summaries,
        sequences=tuple(sequence_summaries),
    )


def _validate_source_output_separation(source: Path, output_dir: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        source_root = source.expanduser().resolve()
        output_root = output_dir.expanduser().resolve()
        if output_root == source_root or source_root in output_root.parents:
            raise ValueError("output directory must not alias or be nested in a proposal source")


def _parse_source_spec(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("proposal sources must use NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def write_summary(summary: ProposalBankFusionSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal",
        action="append",
        type=_parse_source_spec,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args(argv)
    sources: dict[str, Path] = {}
    for name, path in args.proposal:
        if name in sources:
            parser.error(f"duplicate proposal source name: {name}")
        sources[name] = path
    summary = fuse_proposal_banks(
        sources,
        args.output_dir,
        sequences=args.sequences,
    )
    if args.output_json is not None:
        write_summary(summary, args.output_json)
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
