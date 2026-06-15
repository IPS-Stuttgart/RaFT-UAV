#!/usr/bin/env python
"""Extract supported MMUAD ROS bag topics into normalized CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.native_ros import extract_native_rosbag_topic_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="extract supported ROS bag topics for the MMUAD adapter"
    )
    parser.add_argument("--bag-path", type=Path, required=True)
    parser.add_argument("--topic-map-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    args = parser.parse_args(argv)
    result = extract_native_rosbag_topic_map(
        bag_path=args.bag_path,
        topic_map_json=args.topic_map_json,
        output_dir=args.output_dir,
        voxel_size_m=args.voxel_size_m,
        min_points=args.min_cluster_points,
    )
    print("mmuad_native_ros_extraction=ok")
    print(f"candidate_rows={result.manifest['candidate_rows']}")
    print(f"truth_rows={result.manifest['truth_rows']}")
    print(f"image_timestamp_rows={result.manifest.get('image_timestamp_rows', 0)}")
    if "image_timestamp_template_csv" in result.manifest:
        print(f"image_timestamp_template_csv={result.manifest['image_timestamp_template_csv']}")
    print(f"manifest_json={args.output_dir / 'native_ros_extraction_manifest.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
