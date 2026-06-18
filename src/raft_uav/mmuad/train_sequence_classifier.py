"""Train persistable MMUAD sequence-level UAV classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raft_uav.mmuad.classification import (
    SEQUENCE_CLASSIFIER_METHODS,
    load_sequence_class_labels,
    save_sequence_classifier_model,
    sequence_features_from_sequence_root,
    train_sequence_classifier_model,
)
from raft_uav.mmuad.schema import load_jsonable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-train-sequence-classifier",
        description="train a persistable sequence-level UAV classifier from an MMUAD root",
    )
    parser.add_argument("sequence_root", type=Path, help="MMUAD training sequence root")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="training sequence labels or official Track 5 reference CSV/ZIP",
    )
    parser.add_argument("--method", choices=SEQUENCE_CLASSIFIER_METHODS, default="random-forest")
    parser.add_argument("--output", type=Path, required=True, help="output .joblib model path")
    parser.add_argument("--feature-report", type=Path, help="CSV of training sequence features")
    parser.add_argument("--predictions-csv", type=Path, help="optional train-set predictions CSV")
    parser.add_argument("--metrics-json", type=Path, help="optional training metrics JSON")
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split-name")
    parser.add_argument("--no-apply-calibration", action="store_true")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument(
        "--radar-azimuth-convention",
        choices=(
            "north-clockwise",
            "east-counterclockwise",
            "east-clockwise",
            "x-forward-left-positive",
        ),
        default="north-clockwise",
    )
    parser.add_argument("--radar-angle-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--radar-polar-range-std-m", type=float, default=2.0)
    parser.add_argument("--radar-polar-angle-std-deg", type=float, default=2.0)
    parser.add_argument("--radar-polar-z-std-m", type=float, default=5.0)
    parser.add_argument("--camera-fixed-depth-m", type=float)
    parser.add_argument("--camera-std-xy-m", type=float, default=5.0)
    parser.add_argument("--camera-std-z-m", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int)
    args = parser.parse_args(argv)

    features = sequence_features_from_sequence_root(
        args.sequence_root,
        sequence_glob=args.sequence_glob,
        split_file=args.split_file,
        split_name=args.split_name,
        apply_calibration=not args.no_apply_calibration,
        voxel_size_m=args.voxel_size_m,
        min_cluster_points=args.min_cluster_points,
        radar_azimuth_convention=args.radar_azimuth_convention,
        radar_angle_unit=args.radar_angle_unit,
        radar_polar_range_std_m=args.radar_polar_range_std_m,
        radar_polar_angle_std_deg=args.radar_polar_angle_std_deg,
        radar_polar_z_std_m=args.radar_polar_z_std_m,
        camera_fixed_depth_m=args.camera_fixed_depth_m,
        camera_std_xy_m=args.camera_std_xy_m,
        camera_std_z_m=args.camera_std_z_m,
    )
    labels = load_sequence_class_labels(args.reference)
    result = train_sequence_classifier_model(
        train_features=features,
        train_labels=labels,
        method=args.method,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    model_path = save_sequence_classifier_model(result.model, args.output)
    paths = {"model": str(model_path)}
    if args.feature_report is not None:
        args.feature_report.parent.mkdir(parents=True, exist_ok=True)
        result.train_features.to_csv(args.feature_report, index=False)
        paths["feature_report"] = str(args.feature_report)
    if args.predictions_csv is not None:
        args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        result.train_predictions.to_csv(args.predictions_csv, index=False)
        paths["predictions_csv"] = str(args.predictions_csv)
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(
            json.dumps(load_jsonable(result.metrics), indent=2),
            encoding="utf-8",
        )
        paths["metrics_json"] = str(args.metrics_json)

    print("mmuad_sequence_classifier_train=ok")
    for key, value in paths.items():
        print(f"{key}={value}")
    print(f"classification_method={result.model['method']}")
    print(f"classification_train_sequences={len(result.model['train_sequences'])}")
    print(f"classification_feature_columns={len(result.model['feature_columns'])}")
    if result.metrics.get("sequence_accuracy") is not None:
        print(f"train_sequence_accuracy={result.metrics['sequence_accuracy']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
