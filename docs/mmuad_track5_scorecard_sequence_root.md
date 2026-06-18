# MMUAD Track 5 scorecard from a public sequence root

The local `raft-uav-mmuad-track5-scorecard` command can validate an official-style
`mmaud_results.csv` / ZIP directly against the timestamp grid discovered from a
public Track 5 sequence root. This is useful for validation/test folders that
expose sensor-frame timestamps but not hidden labels.

Example:

```bash
raft-uav-mmuad-track5-scorecard \
  --results outputs/mmuad_val/ug2_submission.zip \
  --sequence-root data/mmuad_public \
  --split-name val \
  --timestamp-source image \
  --output-json outputs/mmuad_val/track5_scorecard.json \
  --summary-csv outputs/mmuad_val/track5_scorecard.csv \
  --validation-rows-csv outputs/mmuad_val/track5_validation_rows.csv
```

The `--sequence-root` template path discovers sequences with the existing MMUAD
sequence scanner and uses `Image`, `ground_truth`, `lidar_360`, `livox_avia`,
or `radar_enhance_pcl` folders as timestamp sources. Use `--split-name val` or
`--split-name test` for roots arranged as split folders.

The scorecard remains a local preflight check. Without `--truth`, it can verify
upload shape, sequence/timestamp coverage, duplicates, missing timestamps, and
extra predictions. It cannot compute hidden Codabench pose MSE or UAV-type
accuracy for validation/test labels that are not available locally.
