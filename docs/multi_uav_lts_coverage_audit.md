# Multi-UAV LTS coverage audit

`raft-uav-multi-uav-lts-coverage-audit` checks prediction directories or ZIPs
before packaging/upload. When `--sequence-root` points to the image sequence
folder, the audit can also compare prediction frame IDs and detection coverage
against the actual number of image frames.

Useful strict pre-upload command:

```bash
raft-uav-multi-uav-lts-coverage-audit \
  /mnt/lexar4tb/multi_uav_lts/outputs/official_baseline/predictions \
  --template-zip /mnt/lexar4tb/multi_uav_lts/downloads/submission.zip \
  --sequence-root /mnt/lexar4tb/multi_uav_lts/extracted/TestImages \
  --min-detection-frame-fraction 0.05 \
  --output-json prediction_coverage_audit.json \
  --row-csv prediction_coverage_rows.csv \
  --require-ready
```

The optional detection-frame coverage gate is intentionally off by default.
Set it when a run is expected to produce detections across at least some minimum
fraction of frames in every sequence. This helps catch failed shards that write a
valid but nearly empty prediction file.

New row-level fields:

- `expected_frame_count`
- `detected_frame_count`
- `detection_frame_fraction`

New aggregate fields:

- `detection_frame_fraction_min`
- `detection_frame_fraction_mean`
- `low_detection_coverage_file_count`
- `low_detection_coverage_files`
