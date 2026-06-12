# Experimental MMUAD / CVPR UG2+ adapter

This is a first RaFT-UAV++ portability scaffold for the CVPR UG2+ / MMUAD UAV
tracking and pose-estimation setting.  It is not an official challenge
submission implementation.

The adapter consumes normalized candidate detections:

```csv
sequence_id,time_s,source,track_id,x_m,y_m,z_m,std_xy_m,std_z_m,confidence,class_name
seq001,0.00,radar,track7,1.0,2.0,3.0,2.0,4.0,0.9,uav
```

Ground truth uses:

```csv
sequence_id,time_s,x_m,y_m,z_m
seq001,0.00,1.0,2.0,3.0
```

Timestamp columns may be supplied in seconds via `time_s`, `timestamp_s`,
`time`, or `sec`; in larger exported units via `timestamp_ns`, `timestamp_us`,
or `timestamp_ms`; or as ROS-style second/nanosecond pairs such as
`sec,nanosec`.

Candidate and truth rows can also be supplied as JSON row lists, column maps, or
objects containing common keys such as `candidates`, `detections`, `truth`,
`ground_truth`, `rows`, `data`, or `sequences`.

Run with exported detector/cluster candidates:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --candidate-csv data/mmuad_export/candidates.csv \
  --truth-csv data/mmuad_export/truth.csv \
  --output-dir outputs/mmuad_smoke
```

For non-CSV table or trajectory exports, use the format-aware explicit-file
flags. Compact NumPy trajectories use `time_s,x_m,y_m,z_m` column order, while
JSON tables use the same column names and aliases as CSV rows:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --candidate-file data/mmuad_export/trajectory.npy \
  --truth-file data/mmuad_export/truth.npy \
  --output-dir outputs/mmuad_numpy_smoke
```

Or build simple point-cloud cluster candidates from a CSV/TSV/TXT table with
`sequence_id`, `time_s`, `x_m`, `y_m`, `z_m`:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --point-cloud-csv data/mmuad_export/lidar_points.csv \
  --truth-csv data/mmuad_export/truth.csv \
  --output-dir outputs/mmuad_cluster_smoke
```

Implemented in this first patch:

- normalized candidate and truth schemas;
- alias-tolerant CSV loading plus compact TXT/NumPy trajectory exports;
- lightweight voxel connected-component clustering for exported point-cloud rows;
- a simple single-UAV constant-velocity tracker;
- first-selected-candidate bootstrap;
- selected-tracklet updates plus bounded soft-anchor updates for secondary
  candidates;
- truth metrics when ground truth is supplied;
- CSV/JSON output for estimates, selected tracklets, and metrics;
- unit tests with synthetic candidate/truth data.

Still outside this experimental adapter's supported scope:

- official MMUAD raw archive parsing;
- image detector, point-cloud detector, or UAV classifier training;
- official challenge metric reproduction;
- leaderboard upload tooling.

## Incremental features after the first scaffold

The second patch adds infrastructure that is useful once a local MMUAD export is
available, while still avoiding guesses about undocumented raw binary/archive
formats.

### Calibration/extrinsics JSON

A small calibration interchange format is supported:

```json
{
  "world_frame": "leica_world",
  "sensors": {
    "radar": {
      "translation_m": [1.0, 2.0, 0.5],
      "rpy_deg": [0.0, 0.0, 90.0],
      "time_offset_s": -0.012
    },
    "lidar": {
      "translation_m": [0.0, 0.0, 0.0],
      "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  }
}
```

Use it with explicit candidate files:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --candidate-csv data/mmuad_export/radar_candidates.csv \
  --candidate-csv data/mmuad_export/lidar_candidates.csv \
  --calibration-json data/mmuad_export/calibration.json \
  --truth-csv data/mmuad_export/truth.csv \
  --output-dir outputs/mmuad_calibrated_smoke
```

### Sequence-root discovery

A normalized sequence export can be loaded from folders containing files named
`candidates.csv`, `detections.csv`, `candidates.json`, `truth.json`,
`*_candidates.csv`, delimited variants such as `candidates.tsv` or
`detections.txt`, `points.csv`, `points.tsv`, `points.json`, `*_points.txt`,
`*_points.json`, exported polar
radar and camera detection tables such as `radar_polar.tsv` or `radar_polar.json`,
`camera_detections.txt`, or `camera_detections.json`, compact trajectory arrays such as `trajectory.npy` /
`candidates.npz`, exported ROS topic maps such as `topic_map.json`,
`truth.csv`, compact truth arrays such as `truth.npy`, and optionally
`calibration.json`, `camera_info.json`, `intrinsics.json`, or
camera-folder intrinsics such as `cam0/camera_info.json`. It also recognizes one-level split folders and MMUAD-style
modality subfolders such as `livox_avia/<timestamp>.npy`,
`livox_avia/<timestamp>.json`, `livox_avia/<timestamp>.bin` for exported float32 `x,y,z` or
`x,y,z,intensity` point clouds, `ground_truth/<timestamp>.npy`,
`tracking_results/<timestamp>.npy`,
`radar0/detections.csv` with exported polar range/azimuth columns,
`cam0/detections.csv` with exported pixel/depth or bounding-box columns, and
class-label JSON files such as `classes.json` or `class/<timestamp>.json`:
When a generic candidate CSV/TSV/JSON lacks a `source`/`sensor`/`modality`
column, sequence-root loading uses the enclosing modality folder name such as
`tracking_results`, `radar0`, or `cam0` as the source.

```text
data/mmuad_export/
  seq001/
    calibration.json
    radar_candidates.csv
    lidar_points.csv
    truth.csv
  seq002/
    trajectory.npy
    truth.npy
  seq003/
    candidates.tsv
    truth.csv
  seq003_json/
    candidates.json
    truth.json
    classes.json
  seq004/
    lidar_points.tsv
    lidar_points.json
    truth.csv
  seq005/
    topic_map.json
    radar_export.csv
    truth_export.csv
  seq006/
    radar_polar.tsv
    radar_polar.json
    camera_detections.json
    calibration.json
    truth.csv
  val/
    seq007/
      livox_avia/
        1706255054.386069.npy
        1706255054.386070.bin
        1706255054.386071.json
      ground_truth/
        1706255054.386069.npy
      tracking_results/
        1706255054.386069.npy
      radar0/
        detections.csv
        detections.json
      cam0/
        detections.csv
        camera_info.json
      class/
        1706255054.386069.npy
```

Run all discovered sequences with:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --output-dir outputs/mmuad_sequences
```

When a sequence contains a loadable exported topic map, the files referenced by
that map are loaded through their `column_aliases` and are not also loaded as
generic candidate/truth files. Native-only topic maps without exported file
paths remain for the explicit `--rosbag-path --topic-map-json` extraction path.

### Submission/interchange output

The CLI can write a stable single-UAV trajectory CSV/JSON export:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --output-dir outputs/mmuad_sequences \
  --submission-csv outputs/mmuad_sequences/submission.csv \
  --submission-json outputs/mmuad_sequences/submission.json
```

This is **not** claimed to be the official CVPR UG2+ upload schema.  It is a
stable intermediate format for conversion once the official evaluator/submission
format is available.

Still not implemented:

- official raw MMUAD archive parser;
- native camera/radar/Livox packet readers;
- image detector, point-cloud detector, or UAV classifier training;
- official challenge metric/submission reproduction;
- multi-object tracking or ID metrics;
- leaderboard upload tooling.

## Third incremental patch: portability features

The next patch adds several missing-but-safe pieces that do not require guessing
private binary archive internals.

### ASCII PCD/PLY and Point-Cloud Table Exports

In addition to point-cloud CSV/TSV/TXT/JSON row tables, exported ASCII `.pcd`
and `.ply` files can be clustered into candidate centroids. JSON point-cloud
exports may be row lists, column maps, or objects containing `points`,
`point_cloud`, `pointcloud`, `cloud`, `lidar_points`, `livox_points`,
`detections`, `rows`, or `data`. If the file does not contain per-row
sequence/time columns, the sequence is inferred from the parent folder and the
last numeric token in the filename is used as the timestamp.

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --point-cloud-file data/mmuad_export/seq001/lidar_12.50.pcd \
  --truth-csv data/mmuad_export/seq001/truth.csv \
  --output-dir outputs/mmuad_pcd_smoke
```

This is still an exported-file bridge, not a native Livox packet reader.

### Split manifests

Sequence-root mode can be restricted to a split manifest:

```json
{
  "train": ["seq001", "seq002"],
  "val": ["seq003"]
}
```

Run one split with:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --split-file data/mmuad_export/splits.json \
  --split-name val \
  --output-dir outputs/mmuad_val
```

A CSV manifest with columns `sequence_id,split` is also supported. Exported
metadata files may also use JSON sequence-row layouts such as
`{"sequences": [{"sequence_id": "seq001", "split": "val"}]}` or nested split
objects such as `{"splits": {"val": {"sequence_ids": ["seq001"]}}}`. CSV alias
columns including `id`, `name`, `subset`, and `partition` are accepted.
If the sequence root is already arranged as split folders such as
`train/seq001` and `val/seq002`, omit `--split-file` and pass
`--split-name val`; the CLI filters by the top-level folder name.

### Basic multi-object mode

A lightweight greedy MOT backend is available for exported detections with
multiple UAV/object candidates:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --tracker-mode multi-object \
  --mot-max-association-distance-m 15 \
  --output-dir outputs/mmuad_mot_smoke \
  --submission-zip outputs/mmuad_mot_smoke/submission.zip
```

When truth contains `track_id` or `object_id`, the metrics include simple
MOT-style diagnostics: matches, false positives, false negatives, ID switches,
MOTA-like score, MOTP-like 3D distance, precision, and recall.  These are
repository diagnostics, not official UG2+ challenge metrics.

### ZIP submission bundle

The submission helper can now package the stable CSV/JSON trajectory outputs in
one ZIP file.  The ZIP is an interchange bundle, not an official leaderboard
upload format.

Still not implemented:

- official raw MMUAD archive parser;
- native camera/radar/Livox packet readers;
- official UG2+ evaluator/submission reproduction;
- detector or classifier training;
- official leaderboard upload tooling.

## Fourth incremental patch: inspection and evaluation bridge

This patch adds more missing pieces that help move from toy exports toward a
real MMUAD validation workflow without guessing private binary archive formats.

### Layout inspection

Before writing native parsers, inspect an unpacked/exported MMUAD root:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --inspect-root data/mmuad_export \
  --output-dir outputs/mmuad_inspect \
  --layout-report-json outputs/mmuad_inspect/layout.json \
  --layout-report-csv outputs/mmuad_inspect/layout_files.csv
```

The report classifies files as images, point clouds, candidate tables, truth,
class labels, calibration, ROS recordings, and metadata. Candidate/truth/class
tables may be CSV/TSV/TXT, JSON, or compact NumPy exports. It also infers
timestamps from filenames when possible and reports what each sequence is
missing for a tracking smoke test.

### Binary PCD, BIN, and NumPy point clouds

The point-cloud bridge now supports CSV/TSV/TXT/JSON tables, ASCII and binary PCD
files, ASCII PLY files, simple float32 `.bin` files with `x,y,z` or
`x,y,z,intensity` rows, and simple `.npy` / `.npz` point arrays with shape
`(N, >=3)`.  This still is not a native Livox packet reader, but it covers
common exported point-cloud formats used during dataset inspection.

For sequence-root discovery, JSON or NumPy files with point-cloud names such as
`lidar_points.json`, `lidar_points.npy`, `point_cloud.npz`, or `cloud_12.5.npy`
are clustered as point clouds. NumPy files with trajectory names such as `trajectory.npy`,
`candidates.npz`, or `truth.npy` are loaded as compact trajectory tables with
columns `time_s,x_m,y_m,z_m` in that order. This avoids accidentally clustering
already-tracked trajectory exports into a single point-cloud centroid.
Folder-style exports are also supported: files inside `livox_avia`, `lidar`,
`points`, or `point_cloud` folders are clustered as point clouds; files inside
`ground_truth`, `truth`, `gt`, or `labels` folders are loaded as truth; files
inside `tracking_results`, `tracks`, `trajectories`, `detections`, or
`candidates` folders are loaded as candidate trajectories. Per-frame NumPy pose
files may contain only `x_m,y_m,z_m`; their timestamp is inferred from the
filename. Files named `class`, `classes`, `uav_type`, or `category`, and files
inside matching folders, are read as sequence class labels when they are
CSV/TSV/TXT, JSON, or compact NumPy exports. JSON class files may be direct
labels, row lists, or sequence-to-type maps. Loaded class labels are attached to
otherwise-unlabeled candidates; numeric labels are preserved as strings unless
you provide an external class map for official names. This is still a
normalized/exported-data bridge, not a parser for undocumented native packets.

### Auto calibration loader

In addition to JSON, the calibration loader can now read YAML/YML files when
PyYAML is installed, and simple text/CSV files containing one 4x4 transform
matrix. JSON/YAML sensor entries may also provide common matrix aliases such as
`T_sensor_to_world`, `T_camera_to_world`, `extrinsic_matrix`, `transform_matrix`,
or OpenCV-style `{rows, cols, data}` matrices. Unknown official calibration
formats should still be inspected first instead of silently guessed.

### Submission evaluation bridge

Stable RaFT-UAV MMUAD submission CSVs can be evaluated against normalized truth
CSV files with repository-level trajectory diagnostics:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --evaluate-submission-csv outputs/mmuad_run/submission.csv \
  --evaluate-truth-csv data/mmuad_export/seq001/truth.csv \
  --output-dir outputs/mmuad_eval \
  --evaluation-json outputs/mmuad_eval/eval.json
```

Use `--evaluate-truth-file` instead of `--evaluate-truth-csv` for compact
truth exports such as `truth.npy`, `truth.npz`, or delimited text files.

The evaluation reports mean/RMSE/p95/max 3D error, 2D error, ADE/FDE-style
metrics, matched predictions, unmatched predictions, and truth coverage.  It is
not the official UG2+ evaluator.

## Codabench-style packaging and native layout inventory

The layout inspector can also run against a raw or exported MMUAD/UG2+ tree:

```bash
PYTHONPATH=src python scripts/inspect_mmuad_layout.py \
  data/mmuad_raw_or_export \
  --output-json outputs/mmuad_layout_report.json
```

or through the main CLI:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_raw_or_export \
  --inspect-layout-only \
  --output-dir outputs/mmuad_layout \
  --layout-report-json outputs/mmuad_layout/mmuad_layout_report.json
```

The report counts files by category, including candidate tables, point clouds,
images, calibration files, truth/labels, exported/native topic-map JSON files,
and ROS bag/recording files. JSON table exports follow the same naming
convention as sequence discovery: `candidates.json`, `detections.json`,
`truth.json`, and `classes.json` are reported as usable candidate, truth, or
class-label inputs. NumPy files follow the same convention: `truth.npy` counts
as truth, `trajectory.npz` or `candidates.npy` counts as candidate data, and
`lidar_points.npy` or `cloud_12.5.npz` counts as point-cloud data. It also lists
sequence-like folders and recommends the next adapter step. One-level split
folders such as `train/`, `val/`, and `test/` are unwrapped so summaries keep
the actual sequence IDs. Exported topic maps indicate sequence-root inputs,
while native-only topic maps are kept for explicit ROS extraction with
`--rosbag-path --topic-map-json`.

The public UG2+ Codabench instructions require a ZIP containing a single file
named `mmaud_results.csv`. The exact evaluator schema is not bundled here, so
the helper writes a compact trajectory table with columns
`sequence_id,timestamp,x,y,z,uav_type,score` and packages it using the required
filename:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --output-dir outputs/mmuad_val \
  --ug2-class-name Mavic3 \
  --ug2-results-csv outputs/mmuad_val/mmaud_results.csv \
  --ug2-codabench-zip outputs/mmuad_val/ug2_codabench_submission.zip
```

This is closer to challenge packaging than the generic `submission.zip`, but it
is still not a guarantee of official evaluator compatibility. Once the official
README/evaluator is available, adapt `estimates_to_mmaud_results_frame` to the
exact column names and class labels expected by the server.

## ROS Bag Bridge And Local Evaluation

The adapter can inspect ROS bag containers and create a topic-map template
without requiring ROS Python packages at import time:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --rosbag-path data/mmuad_raw/2023-08-24-11-14-34.bag \
  --rosbag-report-json outputs/mmuad_bag_report.json \
  --topic-map-template-json outputs/mmuad_topic_map_template.json \
  --output-dir outputs/mmuad_bag_inspect
```

Edit the generated topic-map JSON so each relevant topic points to a normalized
export file, then run the tracker. CSV/TSV/TXT/JSON table exports can use
`column_aliases`; compact NumPy trajectory exports such as `radar_trajectory.npy`
and `truth.npy` use the same `time_s,x_m,y_m,z_m` convention as the
explicit-file CLI. JSON topic exports may use row lists, column maps, or objects
containing `points`, `point_cloud`, `candidates`, `detections`, `objects`,
`targets`, `measurements`, `returns`, `predictions`, `truth`, `fixes`, `gps`,
`navsatfix`, `poses`, `rows`, or `data`. The template infers native extraction
kinds for common ROS message types
(`pointcloud2_candidate`, `pose_truth`, `odometry_candidate`, and related truth
variants), while the exported-topic loader still accepts those kinds for CSV or
JSON table and NumPy exports. Table exports marked as `pointcloud2_candidate`
are clustered from point rows using the same lightweight point-cloud bridge.
Table exports marked as `radar_polar_candidate` or `polar_radar_candidate`
are converted from range/azimuth rows using the same polar radar bridge as
`--radar-polar-file`.
The template generator maps clearly polar/range-azimuth radar topic names or
message types to this export path.
Table exports marked as `camera_detections_candidate`,
`image_detections_candidate`, or `detection2d_array_candidate` are
back-projected with the same camera detector bridge as
`--camera-detections-file`; provide `camera_calibration_file` in the topic map
or place `camera_info.json` / `intrinsics.json` beside the detection export.
The template generator maps `vision_msgs/msg/Detection2D(Array)` topics to this
export path, but image object detection itself remains an external preprocessing
step.
Table exports marked as `navsatfix_candidate`, `geopoint_candidate`,
`geopose_candidate`, or their `_truth` variants can provide
`latitude`/`longitude`/`altitude` columns and `enu_origin_lla` in the topic map;
the loader projects them into local ENU `x_m,y_m,z_m` coordinates before
tracking or evaluation.

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --topic-map-json outputs/mmuad_topic_map.json \
  --topic-map-base-dir data/mmuad_topic_exports \
  --output-dir outputs/mmuad_topic_map_smoke \
  --ug2-codabench-zip outputs/mmuad_topic_map_smoke/ug2_submission.zip
```

This bridge is intentionally conservative: it inventories bags and loads
normalized table or compact trajectory exports until the exact local MMUAD raw
layout and topic message types are known.

A local evaluator is available for sanity checking `mmaud_results.csv` against
normalized truth. It is not the official Codabench evaluator:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --evaluate-results-csv outputs/mmuad_topic_map_smoke/mmaud_results.csv \
  --evaluate-truth-csv data/mmuad_export/seq001/truth.csv \
  --evaluation-json outputs/mmuad_topic_map_smoke/local_eval.json \
  --evaluation-rows-csv outputs/mmuad_topic_map_smoke/local_eval_rows.csv \
  --output-dir outputs/mmuad_topic_map_smoke
```

Codabench-style ZIP archives written by this adapter can be sanity-checked
directly; the loader reads the contained `mmaud_results.csv`:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --evaluate-results-zip outputs/mmuad_val/ug2_submission.zip \
  --evaluate-truth-csv data/mmuad_export/val_truth.csv \
  --evaluation-json outputs/mmuad_val/local_eval.json \
  --output-dir outputs/mmuad_val
```

## Leaderboard-Style Local Metrics And Completion

The adapter can compute a closer UG2-style local sanity metric. It still is
**not** the official Codabench evaluator, but it reports the two public
leaderboard quantities that matter most for Track 5 style checks:

- `pose_mse_loss_m2`: mean squared 3D position error.
- `uav_type_accuracy`: sequence/type accuracy when truth rows or a class-map
  file provide UAV type labels.

Evaluate an exported `mmaud_results.csv` with optional class labels:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --evaluate-results-csv outputs/mmuad_val/mmaud_results.csv \
  --evaluate-truth-csv data/mmuad_export/val_truth.csv \
  --evaluation-class-map-csv data/mmuad_export/sequence_classes.csv \
  --evaluation-json outputs/mmuad_val/local_eval.json \
  --evaluation-rows-csv outputs/mmuad_val/local_eval_rows.csv \
  --output-dir outputs/mmuad_val
```

Use `--evaluate-results-zip` instead when the result is still packaged as a
Codabench-style archive.

The submission writer also accepts a sequence-to-class map so different
sequences can use different UAV type labels:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --ug2-class-map-csv data/mmuad_export/sequence_classes.csv \
  --ug2-results-csv outputs/mmuad_val/mmaud_results.csv \
  --ug2-codabench-zip outputs/mmuad_val/ug2_submission.zip \
  --output-dir outputs/mmuad_val
```

If the evaluator expects one prediction per ground-truth/template timestamp,
resample a trajectory to those timestamps before packaging:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --complete-results-to-truth-file data/mmuad_export/val_truth.npy \
  --completed-results-csv outputs/mmuad_val/mmaud_results_completed.csv \
  --completed-results-diagnostics-csv outputs/mmuad_val/completion_rows.csv \
  --completed-ug2-codabench-zip outputs/mmuad_val/ug2_completed.zip \
  --output-dir outputs/mmuad_val
```

Completion supports linear interpolation across short gaps and nearest-hold
extrapolation. The legacy `--complete-results-to-truth-csv` flag remains
available for normalized CSV templates. This is useful for validating row
coverage, but it should be reported separately from raw tracker output because
completion policy can affect leaderboard-style MSE.

## Radar, Camera, And Classification Bridges

The adapter includes lightweight bridges for two common exported modalities that
appear in anti-UAV datasets but are not yet parsed from native raw packets.

### Polar Radar Table Exports

Use `--radar-polar-csv` for radar detections exported as CSV/TSV/TXT
range/azimuth rows, or `--radar-polar-file` for JSON table exports:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --radar-polar-csv data/mmuad_export/seq001/radar_polar.csv \
  --radar-azimuth-convention north-clockwise \
  --radar-angle-unit deg \
  --truth-csv data/mmuad_export/seq001/truth.csv \
  --output-dir outputs/mmuad_radar_polar
```

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --radar-polar-file data/mmuad_export/seq001/radar_polar.json \
  --radar-azimuth-convention north-clockwise \
  --radar-angle-unit deg \
  --truth-csv data/mmuad_export/seq001/truth.csv \
  --output-dir outputs/mmuad_radar_polar_json
```

Supported aliases include `range_m`, `azimuth_deg`, `elevation_deg`, `track_id`,
`confidence`, and common variants. JSON radar exports may be row lists, column
maps, or objects containing `radar_polar`, `radar_detections`, `detections`,
`targets`, `objects`, `measurements`, `returns`, `rows`, or `data`.
Coordinates are in the radar/export frame unless a calibration file is applied
later. This is not a native custom radar message parser.

### Camera Detector Table Exports

Use `--camera-detections-csv` for detector CSV/TSV/TXT outputs, or
`--camera-detections-file` for JSON/table exports, with pixel centers or boxes
and metric depth:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --camera-detections-csv data/mmuad_export/seq001/camera_detections.csv \
  --camera-calibration-file data/mmuad_export/seq001/camera_calibration.json \
  --truth-csv data/mmuad_export/seq001/truth.csv \
  --output-dir outputs/mmuad_camera_detections
```

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --camera-detections-file data/mmuad_export/seq001/camera_detections.json \
  --camera-calibration-file data/mmuad_export/seq001/camera_calibration.json \
  --truth-csv data/mmuad_export/seq001/truth.csv \
  --output-dir outputs/mmuad_camera_detections_json
```

`--camera-calibration-file` can be repeated for multi-camera exports, for
example when `cam0/camera_info.json` and `cam1/camera_info.json` live beside
their detection files. The camera calibration file contains intrinsics
(`fx`, `fy`, `cx`, `cy`) and an
optional camera-to-world rigid transform. Intrinsics can also come from common
matrix fields such as `camera_matrix`, `K`/`k`, `P`/`p`, or
`projection_matrix`, including OpenCV-style `{rows, cols, data}` blocks.
Sequence-root mode discovers common camera-only files such as
`camera_info.json`, `intrinsics.json`, and `camera_intrinsics.json`; these can
live at the sequence root or beside detections in camera folders such as
`cam0/camera_info.json`. Folder-scoped single-camera files are matched to that
camera source and merged with any other discovered camera models. These can
be used for back-projection even when they do not contain generic sensor
extrinsics. Detections can provide `u_px`/`v_px` or
`x1,y1,x2,y2` boxes. JSON exports may be row lists, column maps, or objects with
keys such as `camera_detections`, `detections`, `boxes`, `objects`,
`predictions`, `results`, `instances`, `rows`, or `data`. Depth must come from
`depth_m`/`range_m`, or from a fixed fallback via
`--camera-fixed-depth-m`. Compact boxes such as COCO-style
`bbox=[x,y,width,height]` / `bbox_xywh` and explicit `bbox_xyxy` are also
accepted. JSON rows serialized from Detection2D-style messages may keep nested
`header.stamp`, `header.frame_id`, `bbox.center`, and `results.hypothesis`
fields; the loader flattens those common fields before back-projection. This
bridge does not run image object detection; it consumes detector exports.

### Sequence Class Inference

If detector/candidate rows include useful `class_name` values, the CLI can infer
one UAV type per sequence and use it in `mmaud_results.csv`:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --infer-ug2-class-map-from-candidates \
  --inferred-class-map-csv outputs/mmuad_val/inferred_classes.csv \
  --ug2-results-csv outputs/mmuad_val/mmaud_results.csv \
  --ug2-codabench-zip outputs/mmuad_val/ug2_submission.zip \
  --output-dir outputs/mmuad_val
```

Explicit class maps still take precedence over inferred maps. This is a simple
weighted-vote classifier, not a learned UAV type recognition model.

Class maps may be simple CSV files with `sequence_id,uav_type`, alias CSV files
such as `id,type`, plain JSON objects such as `{"seq001": "Mavic3"}`, or
exported row-style JSON such as
`{"sequences": [{"id": "seq001", "type": "Mavic3"}]}`.

## Native ROS And PointCloud2 Bridge

The adapter also includes an optional native ROS extraction path. Normal imports
do not require ROS or `rosbags`; when `rosbags` is installed, supported topics
can be extracted directly from a ROS2 bag directory or compatible bag path:

```bash
PYTHONPATH=src python scripts/extract_mmuad_rosbag_topics.py \
  --bag-path data/mmuad_raw/seq001 \
  --topic-map-json data/mmuad_raw/seq001/topic_map_native.json \
  --output-dir outputs/mmuad_native_ros_seq001
```

The native extractor currently supports `sensor_msgs/msg/PointCloud2` as
`pointcloud2_candidate`, `sensor_msgs/msg/NavSatFix` as `navsatfix_truth` or
`navsatfix_candidate`, `geographic_msgs/msg/GeoPointStamped` as
`geopoint_truth` or `geopoint_candidate`,
`geographic_msgs/msg/GeoPoseStamped` as `geopose_truth` or
`geopose_candidate`, `vision_msgs/msg/Detection3D` as `detection3d_truth` or
`detection3d_candidate`, `vision_msgs/msg/Detection3DArray` as
`detection3d_array_truth` or `detection3d_array_candidate`,
`visualization_msgs/msg/Marker` as `marker_truth` or `marker_candidate`,
`visualization_msgs/msg/MarkerArray` as `marker_array_truth` or
`marker_array_candidate`,
`geometry_msgs/msg/Pose`, `geometry_msgs/msg/PoseStamped`, and
`geometry_msgs/msg/PoseWithCovarianceStamped` as `pose_truth` or
`pose_candidate`,
`geometry_msgs/msg/PoseArray` as `pose_array_truth` or `pose_array_candidate`,
`geometry_msgs/msg/PointStamped` as `point_truth` or `point_candidate`,
`geometry_msgs/msg/TransformStamped` as `transform_truth` or
`transform_candidate`, `tf2_msgs/msg/TFMessage` as `tf_truth` or
`tf_candidate`, `nav_msgs/msg/Path` as `path_truth` or `path_candidate`, and
`nav_msgs/msg/Odometry` as `odometry_truth` or `odometry_candidate`,
`sensor_msgs/msg/MultiDOFJointState` as `multidof_joint_state_truth` or
`multidof_joint_state_candidate`, and
`trajectory_msgs/msg/MultiDOFJointTrajectory` as
`multidof_joint_trajectory_truth` or `multidof_joint_trajectory_candidate`.
TFMessage
topic-map entries can include `child_frame_id` or `frame_id` to select the UAV
transform from a shared `/tf` stream; Detection3D, Marker, Path, and PoseArray
entries can use `frame_id`, with Detection3DArray, MarkerArray, and PoseArray
rows inheriting the parent message timestamp and frame. MultiDOF rows inherit
the parent frame and use `joint_names` as row provenance and default track IDs
when available. Geodetic topics must include `enu_origin_lla` as `LAT,LON,ALT`
or separate `origin_latitude_deg`, `origin_longitude_deg`, and
`origin_altitude_m` fields so the rows can be projected into local ENU meters.

The CLI can run native extraction and tracking in one step:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --rosbag-path data/mmuad_raw/seq001 \
  --topic-map-json data/mmuad_raw/seq001/topic_map_native.json \
  --native-ros-extract-output-dir outputs/mmuad_native_ros_seq001/extracted \
  --output-dir outputs/mmuad_native_ros_seq001/tracking
```

This is still not a complete official raw MMUAD parser. It is a first native
message bridge for common ROS message types. Custom radar messages, camera
image detectors, and the official evaluator still need dataset-specific work.
