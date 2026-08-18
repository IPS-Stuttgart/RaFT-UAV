# Multi-UAV LTS HOTA-aware gap tubes

This experiment adds a conservative post-processing candidate for short internal
track gaps. It is designed for the organizer-compatible Multi-UAV LTS score,
whose exported HOTA value uses the first TrackEval localization threshold at
IoU 0.05. The method must still pass the existing MOTA, IDF1, canonical HOTA,
and worst-scenario guards before it is considered for a submission.

## Method

For each identity, the post-processor considers only gaps bracketed by two
observed boxes. It:

1. interpolates center coordinates and log box dimensions;
2. expands only the synthetic rows, never detector or tracker observations;
3. widens the tube according to disagreement with the incoming and outgoing
   local velocities;
4. caps the expansion by `--max-scale`;
5. decays synthetic confidence with distance from an observation; and
6. rejects a synthetic row when it overlaps a different identity beyond the
   selected conflict threshold.

The output therefore remains an ordinary LTS prediction directory. No labels,
truth identities, or oracle assignments are consumed.

## Run a candidate

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.hota_tube \
  /path/to/base/predictions \
  --output-dir /path/to/hota_tube/predictions \
  --output-json /path/to/hota_tube/summary.json \
  --max-gap 1 \
  --base-inflation 0.75 \
  --velocity-inflation 0.5 \
  --max-scale 3.0 \
  --conflict-iou 0.3 \
  --confidence-decay 0.9
```

The defaults deliberately target one-frame gaps. A small held-out grid can test:

```text
max_gap:             1, 2
base_inflation:      0.25, 0.5, 0.75, 1.0
velocity_inflation:  0, 0.25, 0.5, 1.0
max_scale:           1.5, 2.0, 3.0
conflict_iou:        0.1, 0.3, 0.5
```

## Evidence requirement

Compare the transformed directory with its unmodified source through the
organizer-compatible evaluator and guarded tournament. Promote it only when the
paired HOTA gain clears the configured confidence bound while MOTA, IDF1, and
worst-scenario performance remain within their floors. Canonical HOTA and LocA
should be retained in the report to expose gains caused by excessively coarse
boxes.
