# Multi-UAV LTS upload QA checklist

Use this checklist before uploading a Multi-UAV LTS ZIP to Codabench. It is
intended to catch local packaging and shard-completion problems before spending
an official submission attempt.

Before generating predictions, apply the deterministic baseline corrections in
[`multi_uav_lts_upstream_fixes.md`](multi_uav_lts_upstream_fixes.md). For
organizer-compatible training evaluation, first-frame population auditing,
seeded identity post-processing, and parameter selection, follow
[`multi_uav_lts_result_improvement.md`](multi_uav_lts_result_improvement.md).
The optional closed-world identity-bank experiment and its guarded raw fallback
are documented in
[`multi_uav_lts_closed_world.md`](multi_uav_lts_closed_world.md). Complete all
training-side selection before starting this final upload checklist.

## 1. Check the source data inventory

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.cli inventory \
  /mnt/lexar4tb/multi_uav_lts/downloads/Test.zip \
  --output-json outputs/multi_uav_lts_test_inventory.json
```

Confirm that the extracted test image folder has the expected sequence count
and that the corrected `BB2P_02` archive has already been applied.

## 2. Audit raw per-sequence text outputs before packaging

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
raft-uav-multi-uav-lts-coverage-audit \
  /mnt/lexar4tb/multi_uav_lts/outputs/official_baseline_via_first_init/predictions \
  --template-zip /mnt/lexar4tb/multi_uav_lts/downloads/submission.zip \
  --sequence-root /mnt/lexar4tb/multi_uav_lts/extracted/TestImages \
  --output-json prediction_coverage_audit.json \
  --row-csv prediction_coverage_rows.csv \
  --require-ready
```

The audit should report:

- zero missing files;
- zero extra files;
- zero empty expected files;
- zero parse errors;
- zero invalid geometry or confidence rows;
- zero out-of-range frame rows;
- zero duplicate frame/object rows.

Investigate any non-empty `blocking_reasons` entry before packaging.

## 3. Package with normalization and sorted rows

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.cli package-submission \
  /mnt/lexar4tb/multi_uav_lts/outputs/official_baseline_via_first_init/predictions \
  --template-zip /mnt/lexar4tb/multi_uav_lts/downloads/submission.zip \
  --output-zip submission.zip \
  --normalize \
  --sort-rows \
  --output-json submission_validation.json \
  --file-summary-csv submission_file_summary.csv
```

Do not upload a ZIP unless the local validation JSON reports the expected file
count and no blocking format errors.

## 4. Re-validate the final ZIP

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.cli validate-submission \
  submission.zip \
  --template-zip /mnt/lexar4tb/multi_uav_lts/downloads/submission.zip \
  --output-json submission_validation_final.json \
  --file-summary-csv submission_file_summary_final.csv
```

The final validation should match the packaged validation. Keep both JSON files
with the uploaded ZIP so the run can be reproduced later.

## 5. Record provenance

Store the following next to the final upload artifact:

- repository commit SHA;
- runner command and environment overrides;
- source prediction folder;
- template ZIP path and checksum;
- final ZIP checksum;
- population audit and selected cross-validation row;
- fixed-population or closed-world summary, including the raw-control deltas;
- upstream-fix summary;
- coverage audit JSON;
- final validation JSON.

This keeps leaderboard results traceable to the exact local artifact that was
uploaded.
