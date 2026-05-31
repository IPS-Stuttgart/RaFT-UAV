# Top-k tracklet graph + weak-z smoother

This method is a raw-stream SOTA candidate, not part of the frozen Opt2
Table-II artifact/proxy branch.

The intent is to convert the artifact audit lessons into a general tracking
row:

1. Build continuous Fortem tracklets from raw radar candidates.
2. Score tracklets using only radar metadata and motion plausibility.
3. Keep top-k globally consistent radar paths.
4. Replay each path with a weak-z radar covariance and fixed-lag smoothing.
5. Soft-weight RF updates by truth-free RF/radar path consistency.
6. Choose the final path by sequence cost plus innovation consistency, not by
   truth error.

Run one flight:

```bash
raft-uav-topk-weakz-tracklet data/raw/AADM2025Dryad \
  --flight Opt2 \
  --variant auto \
  --top-k-paths 8 \
  --beam-width 64 \
  --weakz-radar-xy-std-m 360 \
  --weakz-radar-z-std-m 20000 \
  --acceleration-std 14 \
  --smoother fixed-lag \
  --smoother-lag-s 15 \
  --smoother-acceleration-std 28 \
  --output-dir outputs/topk-weakz-tracklet
```

Important outputs under `outputs/topk-weakz-tracklet/<flight>/`:

- `estimates.csv` - final filtered/smoothed posterior records.
- `filtered_estimates.csv` - pre-smoothing posterior records.
- `selected_radar.csv` - radar rows from the selected top-k path.
- `attempted_selected_radar.csv` - radar rows from all top-k paths.
- `path_diagnostics.csv` - path costs and innovation-consistency scores.
- `tracklet_diagnostics.csv` - tracklet-level features/costs.
- `metrics.json` - paper-sample 2D/3D errors and selected path summary.

Suggested first comparison:

```bash
raft-uav run-baseline data/raw/AADM2025Dryad \
  --flight Opt2 \
  --radar-association tracklet-viterbi \
  --tracklet-variant range-covariance \
  --smoother fixed-lag \
  --smoother-lag-s 20

raft-uav-topk-weakz-tracklet data/raw/AADM2025Dryad \
  --flight Opt2 \
  --top-k-paths 8
```

For a publishable methods result, evaluate this row with leave-one-flight-out
or nested tuning across Opt1/Opt2/Opt3.  Do not tune it against Table-II artifact
counts or artifact proxy means.
