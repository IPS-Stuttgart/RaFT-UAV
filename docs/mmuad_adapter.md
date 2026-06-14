# Experimental MMUAD / CVPR UG2+ adapter

This is a first RaFT-UAV++ portability scaffold for the CVPR UG2+ / MMUAD UAV
tracking and pose-estimation setting.  It supports the public Track 5 folder
layout and upload CSV/ZIP shape, but it is not a direct authenticated
Codabench uploader or a clone of the closed challenge evaluator.

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

Candidate and truth rows can also be supplied as JSON row lists, JSONL/NDJSON
row streams, column maps, or objects containing common keys such as
`candidates`, `detections`, `truth`, `ground_truth`, `rows`, `data`, or
`sequences`. CSV/TSV/TXT/JSON/JSONL table exports may also be gzip-compressed,
for example `candidates.csv.gz` or `truth.jsonl.gz`.
JSON rows exported from common ROS position messages may keep nested
`header.stamp`, `header.frame_id`, `child_frame_id`, `pose.position`,
`pose.pose.position`, `point`, `transform.translation`, or simple `[x,y,z]`
coordinate arrays; the table reader flattens those fields into the normalized
`time_s`, `source`, `track_id`, and `x_m,y_m,z_m` columns.
Flattened table columns such as `header.frame_id`, `child_frame_id`, and
`pose.pose.position.x` are accepted too. If no explicit `source` is provided,
`frame_id` / `header.frame_id` is used as the candidate source; an explicit
CLI, sequence-root, or topic-map source still overrides that frame hint.

Run with exported detector/cluster candidates:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --candidate-csv data/mmuad_export/candidates.csv \
  --truth-csv data/mmuad_export/truth.csv \
  --output-dir outputs/mmuad_smoke
```

For non-CSV table or trajectory exports, use the format-aware explicit-file
flags. Compact NumPy trajectories use `time_s,x_m,y_m,z_m` column order, while
JSON and JSONL tables use the same column names and aliases as CSV rows, and
may use a `.gz` suffix when compressed:

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

- image detector, point-cloud detector, or UAV classifier training;
- closed-server evaluator equivalence beyond the public Track 5 MSE and
  classification-accuracy metrics;
- direct authenticated leaderboard upload tooling.

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
`candidates.csv`, `detections.csv`, `candidates.json`, `candidates.jsonl`,
`truth.json`, `truth.jsonl.gz`, `*_candidates.csv`, delimited variants such as
`candidates.tsv` or `detections.txt`, compressed tables such as
`candidates.csv.gz`, `points.csv`, `points.tsv`, `points.json`,
`points.jsonl`, `points.jsonl.gz`, `*_points.txt`, `*_points.json`,
`*_points.jsonl`, exported polar radar and camera detection tables such as
`radar_polar.tsv` or `radar_polar.json`, `camera_detections.txt`, or
`camera_detections.json`,
compact trajectory arrays such as `trajectory.npy` / `candidates.npz`,
including per-frame `x,y,z` arrays saved as flat vectors or singleton-dimension
column/row vectors,
headerless compact text trajectory frames such as
`ground_truth/<timestamp>.txt` or `tracking_results/<timestamp>.txt` containing
either `x y z` or `time x y z` numeric rows,
exported ROS topic maps such as `topic_map.json` or `topic_map.yaml`,
`truth.csv`, compact truth arrays such as `truth.npy`, and optionally
`calibration.json`, `calibration.yaml`, `camera_info.json`,
`camera_info.yaml`, `intrinsics.json`, `intrinsics.yml`, or
camera-folder intrinsics such as `cam0/camera_info.yaml`. It also recognizes
split/scenario grouping folders and MMUAD-style modality subfolders such as
`livox_avia/<timestamp>.npy`,
`livox_avia/<timestamp>.json`, `livox_avia/<timestamp>.bin` for exported float32 `x,y,z` or
`x,y,z,intensity` point clouds, `ground_truth/<timestamp>.npy`,
`tracking_results/<timestamp>.npy`,
wrapped layouts such as `sensors/livox_avia/stream0/<timestamp>.npy`,
`labels/ground_truth/leica/<timestamp>.npy`, and
`outputs/tracking_results/fused/<timestamp>.npy`,
`radar0/detections.csv` with exported polar range/azimuth columns,
`cam0/detections.csv` with exported pixel/depth or bounding-box columns, and
class-label JSON/YAML files such as `classes.json`, `classes.yaml`, or
`class/<timestamp>.json`:
When a generic candidate CSV/TSV/JSON lacks a `source`/`sensor`/`modality`
column, sequence-root loading uses the enclosing modality folder name such as
`tracking_results`, `livox_avia`, `radar0`, or `cam0` as the source.

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
generic candidate/truth files. Native topic maps without exported file paths are
also discovered when the same sequence folder contains exactly one ROS bag or
recording file, and sequence-root mode runs the native extraction bridge before
tracking. Native topic-map templates without a recording are still inspection
artifacts; use explicit `--rosbag-path --topic-map-file` when a folder contains
multiple recordings or multiple native maps (`--topic-map-json` remains accepted
for existing scripts).

### Submission/interchange output

The CLI can write a stable single-UAV trajectory CSV/JSON export:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --output-dir outputs/mmuad_sequences \
  --submission-csv outputs/mmuad_sequences/submission.csv \
  --submission-json outputs/mmuad_sequences/submission.json
```

This is **not** the official CVPR UG2+ upload schema.  It remains a stable
intermediate interchange format; use `--ug2-official-results-csv` and
`--ug2-official-codabench-zip` for the public Track 5 upload columns.

Still not implemented:

- arbitrary undocumented native camera/radar/Livox binary packet readers;
- image detector, point-cloud detector, or UAV classifier training;
- closed-server evaluator equivalence beyond the public Track 5 MSE and
  classification-accuracy quantities;
- multi-object tracking or ID metrics;
- direct authenticated leaderboard upload tooling.

## Third incremental patch: portability features

The next patch adds several missing-but-safe pieces that do not require guessing
private binary archive internals.

### PCD/PLY and Point-Cloud Table Exports

In addition to point-cloud CSV/TSV/TXT/JSON/JSONL row tables, including
gzip-compressed variants, exported ASCII/binary `.pcd`/`.pcd.gz` and
`.ply`/`.ply.gz` files can be clustered into candidate centroids. JSON point-cloud
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

This is still an exported-file bridge for point-cloud files. Native ROS Livox
CustomMsg topics are handled by the native extraction path described below;
undocumented raw Livox binary packets still need dataset-specific decoding.

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

A CSV manifest with columns `sequence_id,split` is also supported. JSON and
YAML metadata files may also use sequence-row layouts such as
`{"sequences": [{"sequence_id": "seq001", "split": "val"}]}` or nested split
objects such as `{"splits": {"val": {"sequence_ids": ["seq001"]}}}`. CSV alias
columns including `id`, `name`, `subset`, and `partition` are accepted.
Manifest sequence entries may be bare IDs such as `seq001` or split-relative
paths such as `val/seq001`; path entries are useful when split folders reuse the
same sequence leaf name.
If the sequence root is already arranged as split folders such as
`train/seq001`, `val/seq002`, or nested groupings like
`val/fog/seq003`, omit `--split-file` and pass `--split-name val`; the CLI
filters by the top-level folder name.

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
one ZIP file.  The ZIP is an interchange bundle; use the official Track 5 flags
described below for `mmaud_results.csv` upload packaging.

Still not implemented:

- arbitrary undocumented native camera/radar/Livox binary packet readers;
- closed-server evaluator equivalence beyond the public Track 5 MSE and
  classification-accuracy quantities;
- detector or classifier training;
- direct authenticated leaderboard upload tooling.

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

The report classifies files as images, audio recordings, point clouds,
candidate tables, truth, class labels, calibration, ROS recordings, and
metadata. Candidate/truth/class tables may be CSV/TSV/TXT, JSON, or compact
NumPy exports. It also infers timestamps from filenames when possible and
reports what each sequence is missing for a tracking smoke test. Point-cloud
inventory includes common PCD, PLY, LAS/LAZ, and simple float32 `.bin` exports.
Audio inventory covers common WAV/FLAC/AAC/MP3 recordings for evidence
gathering; acoustic detections still need to be exported as candidate tables
before tracking.

### Binary PCD, PLY, LAS, BIN, and NumPy point clouds

The point-cloud bridge now supports CSV/TSV/TXT/JSON/JSONL tables, ASCII,
binary, and `binary_compressed` PCD files, ASCII and binary PLY files,
uncompressed LAS files, optional LASzip/LAZ files through the `pointcloud`
extra, simple float32 `.bin` files with `x,y,z` or `x,y,z,intensity` rows, and
simple `.npy` / `.npz` point arrays with shape `(N, >=3)`. Gzip-compressed
exported files such as `.pcd.gz`, `.ply.gz`, `.las.gz`, and `.bin.gz` are
accepted for those byte/text point-cloud formats. Native ROS Livox CustomMsg
topics are handled by explicit native extraction; undocumented raw Livox binary
packets still need dataset-specific decoding.

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
a local evaluator, not a copy of Codabench's closed execution environment.

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
images, calibration files, truth/labels, exported/native topic-map JSON/YAML files,
and ROS bag/recording files. JSON table exports follow the same naming
convention as sequence discovery: `candidates.json`, `detections.json`,
`truth.json`, and `classes.json` are reported as usable candidate, truth, or
class-label inputs; YAML class-label files such as `classes.yaml` are also
classified as class metadata. NumPy files follow the same convention: `truth.npy` counts
as truth, `trajectory.npz` or `candidates.npy` counts as candidate data, and
`lidar_points.npy` or `cloud_12.5.npz` counts as point-cloud data. It also lists
sequence-like folders and recommends the next adapter step. One-level split
folders such as `train/`, `val/`, and `test/` are unwrapped so summaries keep
the actual sequence IDs. Exported topic maps indicate sequence-root inputs,
while native topic maps paired with exactly one ROS recording can be run from
sequence-root mode through the native extraction bridge. Ambiguous native
folders with multiple maps or recordings should still use explicit
`--rosbag-path` and `--topic-map-file/--topic-map-json`.

The public UG2+ Track 5 README requires a ZIP containing only
`mmaud_results.csv`. The official public CSV columns are
`Sequence,Timestamp,Position,Classification`, where `Position` is written as a
compact `(x,y,z)` tuple string and `Classification` must be an integer UAV type
id. Use `--ug2-official-results-csv` and `--ug2-official-codabench-zip` for
this upload shape:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --output-dir outputs/mmuad_val \
  --ug2-class-map-file data/mmuad_export/sequence_class_ids.csv \
  --ug2-official-results-csv outputs/mmuad_val/mmaud_results.csv \
  --ug2-official-codabench-zip outputs/mmuad_val/ug2_codabench_submission.zip
```

The class-map file used for official output should map each sequence to the
integer challenge class id, for example `sequence_id,uav_type` with values such
as `seq1,0`. If no per-sequence numeric class map is provided, the CLI uses
`--ug2-official-classification` as the default id.

For leaderboard-style packaging, the public README asks for positions at the
given sequence timestamps. With `--sequence-root`, the CLI can resample the
official output to timestamps discovered from Track 5 sequence folders before
writing the official CSV/ZIP:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --split-name val \
  --output-dir outputs/mmuad_val \
  --ug2-official-complete-to-sequence-timestamps \
  --ug2-official-timestamp-source ground-truth-or-all \
  --completion-max-interpolation-gap-s 1.0 \
  --ug2-class-map-file data/mmuad_export/sequence_class_ids.csv \
  --ug2-official-results-csv outputs/mmuad_val/mmaud_results.csv \
  --ug2-official-codabench-zip outputs/mmuad_val/ug2_codabench_submission.zip \
  --ug2-official-validate-on-write
```

`ground-truth-or-all` uses `ground_truth/**/<timestamp>.*` when labels are
present and falls back to the union of public sensor-frame timestamps otherwise.
Official timestamp discovery recurses below the selected modality folders, so
layouts such as `Image/front_camera/<timestamp>.png` or
`livox_avia/stream0/<timestamp>.npy` are included. If frame filenames are not
timestamps, discovery also reads conservative timestamp sidecars such as
`timestamps.csv`, `timestamps.json`, `frame_times.txt`, `timestamps.npy`,
`frame_times.npz`, or `frames.csv/json` inside the same official modality folders.
Table sidecars may use common
timestamp columns such as `time_s`, `timestamp`, `timestamp_ns`,
`timestamp_us`, or `timestamp_ms`; simple text frame lists use the last numeric
token on each non-comment line as seconds. NumPy timestamp sidecars may be 1D
timestamp arrays, or 2D arrays where the last column is the timestamp.
Use `image`, `lidar-360`, `livox-avia`, `radar-enhance-pcl`, or
`all-modalities` when a specific timestamp source is required. The CLI writes
`mmuad_official_timestamp_completion_rows.csv` and
`mmuad_official_timestamp_completion_summary.json` so interpolation and
nearest-hold choices remain auditable, including per-sequence requested,
completed, and dropped timestamp counts. With
`--ug2-official-validate-on-write`, the run also writes
`mmuad_official_submission_validation.json` and
`mmuad_official_submission_validation_rows.csv`, plus
`mmuad_official_upload_manifest.json`, then exits nonzero if the fresh ZIP is
not upload-ready under the local Track 5 preflight checks.
For explicit-file, topic-map, or native ROS runs that do not have a
`--sequence-root`, `--ug2-official-complete-to-sequence-timestamps` can instead
use `--official-validation-template-file` or
`--official-validation-template-csv` as the requested timestamp template.
When normalized truth is loaded through `--truth-csv`, `--truth-file`, a
topic-map truth export, or native ROS extraction, validate-on-write uses those
truth timestamps as the requested official grid.

Before manual upload, validate the official ZIP structure and timestamp
coverage:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --validate-ug2-official-codabench-zip outputs/mmuad_val/ug2_codabench_submission.zip \
  --sequence-root data/mmuad_export \
  --split-name val \
  --ug2-official-timestamp-source ground-truth-or-all \
  --official-validation-json outputs/mmuad_val/official_submission_validation.json \
  --official-validation-rows-csv outputs/mmuad_val/official_submission_validation_rows.csv \
  --official-upload-manifest-json outputs/mmuad_val/official_upload_manifest.json \
  --output-dir outputs/mmuad_val
```

This preflight check enforces a ZIP containing only a root-level
`mmaud_results.csv` file, exact
`Sequence,Timestamp,Position,Classification` columns, nonblank/non-missing
sequence IDs, finite timestamps and `(x,y,z)` position tuples, integer class
IDs, duplicate prediction detection, and optional timestamp coverage against a
truth/template file or Track 5 sequence-root folders. It exits nonzero when the
package is not upload-ready, which makes it suitable for local CI before a
manual Codabench upload. A ZIP such as `submission/mmaud_results.csv` is
rejected because the result file is not at the archive root.
The validation JSON separates structural validity from leaderboard readiness:
`valid` may be true for a well-formed ZIP even when no timestamp template was
available, while `score_valid_for_leaderboard`, `leaderboard_ready`, and
`codabench_upload_ready` only become true after timestamp coverage has been
checked and no invalid, duplicate, missing, or extra prediction rows remain.
When readiness is false, `leaderboard_blocking_reasons` records the concrete
preflight gaps, including `timestamp_template_not_checked` when no template was
provided and `no_template_timestamps` when the template has no usable requested
timestamps. The validation summary also includes per-sequence readiness under
`sequences`, so missing template timestamps and prediction-only extra sequence
IDs can be diagnosed without scanning every validation row. Rows with blank,
missing, or missing-like official `Sequence` values are grouped under the
reserved `__invalid_sequence__` summary key. With a nonzero timestamp tolerance,
only the nearest unused prediction covers a requested template timestamp; other
non-duplicate predictions in the same tolerance window remain extras.
`mmuad_official_upload_manifest.json` is a compact, machine-readable readiness
index for release scripts and CI. It records the checked artifact path,
validation JSON/row paths, ZIP members and columns, global readiness flags,
blocking reasons, sequence counts, and per-sequence prediction/template
coverage. It does not upload results or replace the closed Codabench evaluator.
Official Track 5 CSV/ZIP writers also fail before writing when tracker output
contains non-finite timestamps or positions, preventing a seemingly valid ZIP
from silently dropping rows. For local diagnostics only, pass
`--ug2-official-invalid-row-policy drop` to keep the older filtering behavior.
`--official-validation-template-file` may point to either a normalized
truth/template file or an official Track 5 CSV/ZIP; for official files, only
`Sequence` and `Timestamp` are used as the requested prediction grid.
Boolean `Classification` values are rejected rather than silently converted to
`0` or `1`.
For compatibility with array-like exports from other Track 5 pipelines, the
local validator/evaluator also accepts `Position` cells formatted as `[x y z]`
or `x y z`, plus NumPy-style wrappers such as `array([x, y, z])` or
`np.array([x y z])`, in addition to comma/semicolon-separated tuple/list
strings.

The older local diagnostic result table with
`sequence_id,timestamp,x,y,z,uav_type,score` remains available for repository
evaluation and completion tools:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --output-dir outputs/mmuad_val \
  --ug2-class-name Mavic3 \
  --ug2-results-csv outputs/mmuad_val/mmaud_results.csv \
  --ug2-codabench-zip outputs/mmuad_val/ug2_codabench_submission.zip
```

The local evaluator accepts either shape, but it is still a transparent local
implementation of the public MSE/classification-accuracy quantities rather than
a copy of Codabench's closed evaluator environment.

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
export file, or provide the same schema as YAML. CSV/TSV/TXT/JSON/JSONL table exports can use
`column_aliases`, and table exports may be gzip-compressed; compact NumPy
trajectory exports such as `radar_trajectory.npy` and `truth.npy` use the same
`time_s,x_m,y_m,z_m` convention as the explicit-file CLI. JSON topic exports
may use row lists, JSONL/NDJSON row
streams, column maps, or objects containing `points`, `point_cloud`,
`candidates`, `detections`, `objects`, `targets`, `measurements`, `returns`,
`predictions`, `truth`, `fixes`, `gps`, `navsatfix`, `poses`, `rows`, or
`data`. ROS-shaped JSON row exports can keep nested `header.stamp`,
`pose.position`, `pose.pose.position`, `point`, or `transform.translation`
fields. Detection3D-style exports can use nested `bbox.center.position` JSON
objects or flattened table columns such as `bbox.center.position.x`; flattened
pose columns such as `pose.pose.position.x` and ROS timestamp pairs such as
`header.stamp.sec` / `header.stamp.nanosec` are also accepted.
Detection result metadata can be nested as `results.hypothesis` or flattened as
`results.0.hypothesis.class_id` / `results.0.hypothesis.score`; these populate
`class_name` and `confidence`. When a container such as a path export has the
timestamp in the parent `header`, that timestamp is propagated to child pose
rows. Explicit topic-map `source` values override any `header.frame_id` source
hint from the row. The template infers native extraction kinds for common ROS
message types
(`pointcloud2_candidate`, `pose_truth`, `odometry_candidate`, and related truth
variants), while the exported-topic loader still accepts those kinds for CSV or
JSON/JSONL table and NumPy exports. Table exports marked as `pointcloud2_candidate`
are clustered from point rows using the same lightweight point-cloud bridge.
Topic-map entries marked as `livox_custommsg_candidate` also use that point-row
bridge for exported tables and, during native ROS extraction, decode common
Livox CustomMsg `points` arrays before clustering.
Table exports marked as `radar_polar_candidate` or `polar_radar_candidate`
are converted from range/azimuth rows using the same polar radar bridge as
`--radar-polar-file`.
For native ROS extraction, the same kinds also accept common custom
range/azimuth message shapes such as `detections`, `targets`, `returns`, or
parallel `ranges`/`azimuths` arrays. Native radar angles default to radians;
set `angle_unit: deg` in the topic map for degree-valued custom messages. The
template generator maps clearly polar/range-azimuth radar topic names or
message types to this export path.
Table exports marked as `camera_detections_candidate`,
`camera_detection_candidate`, `image_detections_candidate`, `detection2d_candidate`,
or `detection2d_array_candidate` are
back-projected with the same camera detector bridge as
`--camera-detections-file`; provide `camera_calibration_file` in the topic map
or place `camera_info.json` / `intrinsics.json` beside the detection export. For
native ROS extraction, a `sensor_msgs/msg/CameraInfo` topic can instead be
listed as `camera_info_calibration` with the same `source` as the Detection2D
topic, and the extractor will use its `K`/`P` intrinsics for back-projection.
The template generator maps `vision_msgs/msg/Detection2D(Array)` topics to this
export path and `sensor_msgs/msg/CameraInfo` topics to
`camera_info_calibration`, but image object detection itself remains an external
preprocessing step.
Table exports marked as `navsatfix_candidate`, `geopoint_candidate`,
`geopose_candidate`, or their `_truth` variants can provide
`latitude`/`longitude`/`altitude` columns and `enu_origin_lla` in the topic map;
the loader projects them into local ENU `x_m,y_m,z_m` coordinates before
tracking or evaluation.

For ROS2 bag directories, the inspection report includes the metadata fields
needed to inventory an unpacked raw sequence before writing parser-specific
topic maps: storage identifier, serialization format, listed relative bag
files, discovered `.db3`/`.mcap` files, total message count, duration, and
starting time when those fields are present in `metadata.yaml`.
Standalone `.db3` and `.mcap` files are also recognized. If the optional
`rosbags` package is installed, inspection lists native topics and the topic-map
template can be generated directly from the recording. Without `rosbags`, the
report remains dependency-safe and records that a native reader is unavailable,
so the next step is still clear: install `rosbags` or export the topics to CSV
and use the topic-map bridge.

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --topic-map-file outputs/mmuad_topic_map.yaml \
  --topic-map-base-dir data/mmuad_topic_exports \
  --output-dir outputs/mmuad_topic_map_smoke \
  --ug2-codabench-zip outputs/mmuad_topic_map_smoke/ug2_submission.zip
```

This bridge is intentionally conservative: it inventories bags and loads
normalized table or compact trajectory exports until the exact local MMUAD raw
layout and topic message types are known.

A local evaluator is available for sanity checking `mmaud_results.csv` against
normalized truth. By default it uses a nearest-time development diagnostic; it
is not the official Codabench runtime:

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

The adapter can compute a closer UG2-style local sanity metric with
`--evaluation-protocol public-track5`. It still is **not** the closed
Codabench evaluator runtime, but it follows the public Track 5 rule shape:
predictions are aligned to the truth/template timestamps, missing/extra/duplicate
predictions are reported, and the matched rows expose the two public leaderboard
quantities:

- `mean_square_loss_m2` / `pose_mse_loss_m2`: mean squared 3D position error.
- `classification_accuracy` / `uav_type_accuracy`: sequence/type accuracy when truth rows or a class-map
  file provide UAV type labels.

Official Track 5 `Classification` cells must be integer IDs in the writer,
validator, and evaluator loader. Integer-like IDs are normalized before
comparison, so `2`, `2.0`, and NumPy/pandas numeric scalars score as the same
class ID. The older local diagnostic `uav_type` table still compares
non-numeric names such as `Mavic3` as stripped strings.

Evaluate a packaged official ZIP with optional class labels:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --evaluate-results-zip outputs/mmuad_val/ug2_official_submission.zip \
  --evaluate-truth-csv data/mmuad_export/val_truth.csv \
  --evaluation-protocol public-track5 \
  --evaluation-timestamp-tolerance-s 1e-6 \
  --evaluation-require-complete-track5 \
  --evaluation-class-map-file data/mmuad_export/sequence_classes.yaml \
  --evaluation-json outputs/mmuad_val/local_eval.json \
  --evaluation-rows-csv outputs/mmuad_val/local_eval_rows.csv \
  --output-dir outputs/mmuad_val
```

Use `--evaluate-results-csv` for local metric checks of an unpacked
`mmaud_results.csv`; a complete CSV can set `score_valid_for_leaderboard=true`,
but it is not Codabench upload-ready until packaged as an official ZIP. The
truth side may be a normalized truth file or an
official Track 5 CSV/ZIP with `Sequence`, `Timestamp`, `Position`, and
`Classification` columns; official truth rows are parsed into local normalized
coordinates for the public Track 5 sanity metric. The summary includes `truth_count`,
`prediction_count`,
`missing_prediction_count`, `extra_prediction_count`,
`duplicate_prediction_count`, `truth_coverage_fraction`, and
`all_truth_timestamps_matched` so a leaderboard-style package can be checked for
timestamp coverage before upload. It also reports `leaderboard_ready`,
`score_valid_for_leaderboard`, `codabench_upload_ready`, and
`leaderboard_blocking_reasons`; `score_valid_for_leaderboard` covers the public
timestamp/classification metric grid, while `leaderboard_ready` also requires an
official upload-ready ZIP. These stay false/nonempty until every requested
truth/template timestamp has exactly one prediction, there are no extras or
duplicates, and the checked artifact is upload-shaped. Timestamp matching is
one-to-one: even when a nonzero tolerance is used, one prediction cannot cover
multiple requested timestamps. Empty truth/template files report
`no_truth_timestamps` and are never treated as leaderboard-ready. Per-sequence
summaries include both requested sequences and prediction-only extra sequences,
so an unexpected sequence ID is visible without inspecting every row. With
`--evaluation-require-complete-track5`, the CLI writes the JSON/row artifacts
and exits nonzero when those strict public Track 5 readiness checks fail.

The submission writer also accepts a sequence-to-class map so different
sequences can use different UAV type labels:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --ug2-class-map-file data/mmuad_export/sequence_classes.yaml \
  --ug2-results-csv outputs/mmuad_val/mmaud_results.csv \
  --ug2-codabench-zip outputs/mmuad_val/ug2_submission.zip \
  --output-dir outputs/mmuad_val
```

The older `--ug2-class-map-csv` and `--evaluation-class-map-csv` option names
remain accepted for existing scripts; the `*-file` aliases are clearer because
class maps may now be CSV, JSON, or YAML.

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

The adapter includes lightweight bridges for two common anti-UAV modalities.
Polar radar can be loaded from exported tables or from common native ROS
range/azimuth message shapes. Camera detections still require exported detector
outputs or native Detection2D messages; raw image object detection remains
external.

### Polar Radar Table Exports

Use `--radar-polar-csv` for radar detections exported as CSV/TSV/TXT
range/azimuth rows, or `--radar-polar-file` for JSON/JSONL table exports,
including `.gz` compressed files:

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
`confidence`, and common variants. JSON/JSONL radar exports may be row lists, column
maps, or objects containing `radar_polar`, `radar_detections`, `detections`,
`targets`, `objects`, `measurements`, `returns`, `rows`, or `data`.
Wrapper-level `sequence_id` and timestamp fields are propagated to nested
detections when individual rows do not already provide them.
Coordinates are in the radar/export frame unless a calibration file is applied
later. Native ROS extraction also accepts topic-map entries marked
`radar_polar_candidate` or `polar_radar_candidate` when messages expose
range/azimuth detections in object arrays such as `detections`, `targets`,
`measurements`, or `returns`, or in parallel arrays such as `ranges` and
`azimuths`. This handles common custom ROS radar shapes, but undocumented binary
payloads still need dataset-specific decoding.

Sequence-root discovery also treats Cartesian radar tables inside radar sensor
folders such as `radar0/detections.csv` or `mmwave/detections.json` as normal
candidate detections when they expose a timestamp plus `x`/`y`/`z` position
columns. Polar range/azimuth tables in the same folders continue through the
polar radar bridge above.

### Camera Detector Table Exports

Use `--camera-detections-csv` for detector CSV/TSV/TXT outputs, or
`--camera-detections-file` for JSON/JSONL table exports, with pixel centers or boxes
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
`camera_info.json`, `camera_info.yaml`, `intrinsics.json`, `intrinsics.yml`,
and `camera_intrinsics.json`; these can
live at the sequence root or beside detections in camera folders such as
`cam0/camera_info.yaml`. Folder-scoped single-camera files are matched to that
camera source and merged with any other discovered camera models. These can
be used for back-projection even when they do not contain generic sensor
extrinsics. Detections can provide `u_px`/`v_px` or
`x1,y1,x2,y2` boxes. JSON/JSONL exports may be row lists, column maps, or objects with
keys such as `camera_detections`, `detections`, `boxes`, `objects`,
`predictions`, `results`, `instances`, `rows`, or `data`. Depth must come from
`depth_m`/`range_m`, or from a fixed fallback via
`--camera-fixed-depth-m`. Compact boxes such as COCO-style
`bbox=[x,y,width,height]` / `bbox_xywh` and explicit `bbox_xyxy` are also
accepted. JSON rows serialized from Detection2D-style messages may keep nested
`header.stamp`, `header.frame_id`, `bbox.center`, and `results.hypothesis`
fields; the loader flattens those common fields before back-projection.
Metric depth may also be carried on `bbox.center.position.z` or `bbox.center.z`
when an export preserves a depth/range estimate with the detection. This
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
such as `id,type`, plain JSON/YAML objects such as `{"seq001": "Mavic3"}`, or
exported row-style JSON/YAML such as
`{"sequences": [{"id": "seq001", "type": "Mavic3"}]}`.

## Native ROS And PointCloud2 Bridge

The adapter also includes an optional native ROS extraction path. Normal imports
do not require ROS or `rosbags`; when `rosbags` is installed, supported topics
can be extracted directly from a ROS2 bag directory or compatible bag path:

```bash
PYTHONPATH=src python scripts/extract_mmuad_rosbag_topics.py \
  --bag-path data/mmuad_raw/seq001 \
  --topic-map-file data/mmuad_raw/seq001/topic_map_native.yaml \
  --output-dir outputs/mmuad_native_ros_seq001
```

The native extractor currently supports `sensor_msgs/msg/PointCloud2` as
`pointcloud2_candidate`, `livox_ros_driver/msg/CustomMsg` and
`livox_ros_driver2/msg/CustomMsg` as `livox_custommsg_candidate`,
`sensor_msgs/msg/CameraInfo` as
`camera_info_calibration` intrinsics for Detection2D back-projection,
common custom polar/range-azimuth radar message shapes as
`radar_polar_candidate` or `polar_radar_candidate`,
`sensor_msgs/msg/NavSatFix` as `navsatfix_truth` or
`navsatfix_candidate`, `geographic_msgs/msg/GeoPointStamped` as
`geopoint_truth` or `geopoint_candidate`,
`geographic_msgs/msg/GeoPoseStamped` as `geopose_truth` or
`geopose_candidate`, `vision_msgs/msg/Detection2D` or
`vision_msgs/msg/Detection2DArray` as `camera_detections_candidate`; legacy
singular aliases such as `camera_detection_candidate` and `detection2d_candidate`
are also accepted when camera calibration and depth/fixed-depth metadata are provided.
Camera calibration can
come from a sidecar calibration/intrinsics file, a nearby `camera_info` file, or
a topic-map `camera_info_calibration` export from a native CameraInfo topic,
`vision_msgs/msg/Detection3D` as `detection3d_truth` or
`detection3d_candidate`, `vision_msgs/msg/Detection3DArray` as
`detection3d_array_truth` or `detection3d_array_candidate`,
`visualization_msgs/msg/Marker` as `marker_truth` or `marker_candidate`,
`visualization_msgs/msg/MarkerArray` as `marker_array_truth` or
`marker_array_candidate`,
`geometry_msgs/msg/Pose`, `geometry_msgs/msg/PoseStamped`, and
`geometry_msgs/msg/PoseWithCovarianceStamped` as `pose_truth` or
`pose_candidate`,
`geometry_msgs/msg/PoseArray` as `pose_array_truth` or `pose_array_candidate`,
`geometry_msgs/msg/Point` and `geometry_msgs/msg/PointStamped` as
`point_truth` or `point_candidate`,
`geometry_msgs/msg/TransformStamped` as `transform_truth` or
`transform_candidate`, `tf2_msgs/msg/TFMessage` as `tf_truth` or
`tf_candidate`, `nav_msgs/msg/Path` as `path_truth` or `path_candidate`, and
`nav_msgs/msg/Odometry` as `odometry_truth` or `odometry_candidate`,
`sensor_msgs/msg/MultiDOFJointState` as `multidof_joint_state_truth` or
`multidof_joint_state_candidate`, and
`trajectory_msgs/msg/MultiDOFJointTrajectory` as
`multidof_joint_trajectory_truth` or `multidof_joint_trajectory_candidate`.
PointCloud2 decoding handles both compact and organized clouds, including
`row_step` padding between rows.
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
  --topic-map-file data/mmuad_raw/seq001/topic_map_native.yaml \
  --native-ros-extract-output-dir outputs/mmuad_native_ros_seq001/extracted \
  --output-dir outputs/mmuad_native_ros_seq001/tracking \
  --submission-csv outputs/mmuad_native_ros_seq001/tracking/submission.csv \
  --ug2-results-csv outputs/mmuad_native_ros_seq001/tracking/mmaud_results.csv \
  --ug2-official-results-csv \
  outputs/mmuad_native_ros_seq001/tracking/official_mmaud_results.csv \
  --ug2-official-codabench-zip \
  outputs/mmuad_native_ros_seq001/tracking/official_submission.zip \
  --ug2-official-validate-on-write
```

Native extraction uses the same tracker artifact writer as exported CSV/sequence
roots, so `submission.csv/json/zip`, UG2 result CSV/ZIP, official Track 5
CSV/ZIP, validation JSON/rows, and trajectory metrics can be requested from the
same run.
Sequence-root mode can also run native extraction automatically when each
sequence folder contains one native `topic_map*.json/yaml` file and one ROS bag
or recording file. Per-sequence native manifests are written below
`native_ros_extracted/<sequence_id>/` by default and summarized in
`native_ros_sequence_manifests.json`; pass `--native-ros-extract-output-dir` to
choose a different extraction base directory.
If native extraction writes a manifest but yields no candidate rows, the CLI
exits before tracking and points to `native_ros_extraction_manifest.json`; update
the topic map to include candidate-bearing topics or export candidate detections
first.

This is still not a complete native ROS parser for every possible MMUAD bag. It
is a first native message bridge for common ROS message types. Undocumented
binary radar/Livox payloads and camera image detectors still need
dataset-specific work.
