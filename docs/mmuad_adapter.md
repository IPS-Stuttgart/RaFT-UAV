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

Run with exported detector/cluster candidates:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --candidate-csv data/mmuad_export/candidates.csv \
  --truth-csv data/mmuad_export/truth.csv \
  --output-dir outputs/mmuad_smoke
```

Or build simple point-cloud cluster candidates from a CSV with `sequence_id`,
`time_s`, `x_m`, `y_m`, `z_m`:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --point-cloud-csv data/mmuad_export/lidar_points.csv \
  --truth-csv data/mmuad_export/truth.csv \
  --output-dir outputs/mmuad_cluster_smoke
```

Implemented in this first patch:

- normalized candidate and truth schemas;
- alias-tolerant CSV loading;
- lightweight voxel connected-component clustering for point-cloud CSV rows;
- a simple single-UAV constant-velocity tracker;
- first-selected-candidate bootstrap;
- selected-tracklet updates plus bounded soft-anchor updates for secondary
  candidates;
- truth metrics when ground truth is supplied;
- CSV/JSON output for estimates, selected tracklets, and metrics;
- unit tests with synthetic candidate/truth data.

Not implemented yet:

- official MMUAD raw archive parsing;
- camera, radar, and LiDAR calibration/extrinsic handling;
- image detector, point-cloud detector, or UAV classifier training;
- official CVPR UG2+ submission file generation;
- official challenge metric reproduction;
- multi-object tracking;
- use of challenge validation/test splits or leaderboard upload tooling.

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
`candidates.csv`, `detections.csv`, `*_candidates.csv`, `points.csv`,
`*_points.csv`, `truth.csv`, and optionally `calibration.json`:

```text
data/mmuad_export/
  seq001/
    calibration.json
    radar_candidates.csv
    lidar_points.csv
    truth.csv
  seq002/
    candidates.csv
    truth.csv
```

Run all discovered sequences with:

```bash
PYTHONPATH=src python -m raft_uav.mmuad.cli \
  --sequence-root data/mmuad_export \
  --output-dir outputs/mmuad_sequences
```

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
