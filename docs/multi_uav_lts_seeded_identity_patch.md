# Multi-UAV LTS explicit first-frame identity seeding

The competition first-frame label files use the row format

```text
frame_id,object_id,x1,y1,width,height,confidence,class_id,visibility
```

The external `YOLOv12-BoT-SORT-ReID` inference script previously parsed column
zero as an object ID and then discarded the parsed value before calling
BoT-SORT. Column zero is the frame ID, so every first-frame row was effectively
read as identity `1`; BoT-SORT then allocated unrelated sequential IDs according
to detector row order.

`raft_uav.multi_uav_lts.upstream_fixes` now patches the external checkout so
that it:

1. reads `object_id` from column one;
2. passes the complete first-frame ID vector into `BoTSORT.update`;
3. accepts that vector only on tracker frame one;
4. rejects missing, duplicate, non-positive, Boolean, fractional, or
   threshold-filtered seed IDs;
5. activates each first-frame track with its supplied identity; and
6. reserves the global allocator above the largest seed ID so later ordinary
   births cannot collide.

The source transformation remains idempotent and is covered by executable
runtime tests. Apply it through the existing command:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.upstream_fixes \
  /mnt/lexar4tb/multi_uav_lts/repos/YOLOv12-BoT-SORT-ReID/BoT-SORT
```

The improved and official-baseline runners already call this patch layer when
upstream fixes are enabled. Existing `.raft-uav-original` backups are preserved.

After applying the patch, regenerate predictions rather than post-processing an
old submission. The fixed-population postprocessor remains useful for fragment
relinking and conservative birth suppression, but it should no longer be needed
to repair the initial identity permutation.
