# Archived sequence roots for the MMUAD Track 5 scorecard

`raft-uav-mmuad-track5-scorecard` can build its timestamp-coverage template from a public Track 5 sequence root.  The command now also accepts ZIP and TAR-family sequence-root archives directly.  The archive is safely extracted before normal sequence discovery runs, and an extraction manifest is written for reproducibility.

Example:

```bash
raft-uav-mmuad-track5-scorecard \
  --results outputs/mmuad_val/ug2_official_submission.zip \
  --truth data/mmuad_export/val_truth.csv \
  --sequence-root data/mmuad_val.zip \
  --split-name val \
  --timestamp-source image \
  --sequence-root-archive-manifest-json outputs/mmuad_val/scorecard_archive_manifest.json \
  --output-json outputs/mmuad_val/track5_scorecard.json \
  --summary-csv outputs/mmuad_val/track5_scorecard.csv \
  --require-leaderboard-ready
```

Useful flags:

- `--sequence-root-archive-extract-dir`: choose the extraction parent directory.  By default, extraction goes beside `--output-json` under `mmuad_scorecard_sequence_root_archive/`.
- `--sequence-root-archive-manifest-json`: choose the manifest path.  By default, the manifest is written beside `--output-json` as `mmuad_scorecard_sequence_root_archive_manifest.json`.

The manifest records the archive format, SHA-256, extraction root, extracted member list, and skipped unsafe members.  The same safe extractor rejects absolute paths, parent-directory traversal, drive-like member names, and archive links.

This is still a local preflight workflow.  It does not upload to Codabench and does not replace the closed challenge evaluator.
