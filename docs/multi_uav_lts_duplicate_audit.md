# Multi-UAV LTS duplicate prediction audit

`python -m raft_uav.multi_uav_lts.duplicate_audit` checks prediction
directories or ZIP files for duplicate `(frame_id, object_id)` keys.

A submission can have valid file names and valid numeric rows while still
containing repeated detections for the same object identifier in the same frame.
Those duplicates are easy to miss in large sharded inference outputs and can
pollute local diagnostics or official uploads.

Example:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.duplicate_audit \
  /mnt/lexar4tb/multi_uav_lts/outputs/official_baseline/predictions \
  --output-json outputs/multi_uav_lts_duplicate_audit.json \
  --file-summary-csv outputs/multi_uav_lts_duplicate_file_summary.csv \
  --duplicate-keys-csv outputs/multi_uav_lts_duplicate_keys.csv \
  --require-clean
```

Outputs:

- `duplicate_audit.json`: aggregate clean flag, duplicate counts, and file list;
- `duplicate_file_summary.csv`: per-file row, parse, duplicate-key, and
  duplicate-row counts;
- `duplicate_keys.csv`: one row per duplicated `(frame_id, object_id)` key.

Use this after shard inference and before packaging/upload. It complements
`raft-uav-multi-uav-lts-coverage-audit`, which focuses on missing, extra, empty,
malformed, unsorted, and frame-bound coverage checks.
