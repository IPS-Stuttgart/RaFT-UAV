# Multi-UAV LTS upstream BoT-SORT fixes

The official baseline runner treats `YOLOv12-BoT-SORT-ReID/BoT-SORT` as an
external checkout. RaFT-UAV therefore does not silently edit that checkout.
Instead, the following command applies a small, validated compatibility patch
before inference:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.upstream_fixes \
  /mnt/lexar4tb/multi_uav_lts/repos/YOLOv12-BoT-SORT-ReID/BoT-SORT \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/upstream_fixes_summary.json
```

By default the command creates sibling backups ending in
`.raft-uav-original`. It is idempotent: a second invocation reports zero
changed files.

Use check mode in automation to verify that the fixes have already been
applied without writing anything:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.upstream_fixes \
  /mnt/lexar4tb/multi_uav_lts/repos/YOLOv12-BoT-SORT-ReID/BoT-SORT \
  --check
```

Check mode returns nonzero when an update is required.

## Applied corrections

The patch performs four narrowly scoped corrections:

1. Empty detector frames call `tracker.update` with an empty detection array.
   This advances the Kalman prediction, lost-track ages, and track buffer rather
   than compressing time across detection gaps.
2. `--fuse-score` receives positive semantics and defaults to enabled, matching
   the effective behavior of the original code. `--no-fuse-score` explicitly
   disables it. The legacy `mot20` tracker field is still populated so the
   tracker implementation and other upstream tools remain compatible.
3. Only activated tracks are emitted. First-frame seeds remain immediately
   active, while later one-frame false tracks are not written before temporal
   confirmation.
4. Output box coordinates retain floating-point precision instead of being
   rounded to two decimal places before RaFT-UAV packaging.

The patcher verifies every expected source block before writing. If the
external repository changes incompatibly, it raises an error rather than
partially modifying the checkout. Inspect the upstream revision and update the
patch rules deliberately in that case.

After applying the patch, run the baseline normally and then follow
[`multi_uav_lts_result_improvement.md`](multi_uav_lts_result_improvement.md)
for HOTA evaluation and first-frame-seeded post-processing.
