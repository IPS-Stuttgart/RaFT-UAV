# PyRecEst hypothesis replay ranking migration

RaFT-UAV's `global-tracklet` branch should keep AERPAW/Fortem-specific segment
construction local, but delegate top-k replay ranking to PyRecEst.

Use:

```python
from raft_uav.baselines.pyrecest_hypothesis_ranking import (
    GlobalTrackletHypothesisRankingConfig,
    rank_global_tracklet_replays,
)

ranked_rows = rank_global_tracklet_replays(
    path_replay_rows,
    config=GlobalTrackletHypothesisRankingConfig(
        graph_cost_weight=1.0,
        replay_nis_weight=args.global_tracklet_replay_nis_weight,
        residual_weight=args.global_tracklet_replay_residual_weight,
        residual_clip_m=args.global_tracklet_residual_clip_m,
        unsupported_rf_weight=args.global_tracklet_unsupported_rf_weight,
        hard_quarantine_weight=args.global_tracklet_hard_quarantine_weight,
    ),
)
```

Each `path_replay_rows` entry should contain `path_id`, `graph_cost`, `records`,
`track_switches`, `missed_radar_count`, `unsupported_rf_count`,
`hard_quarantined_segments_used`, `tail_duration_s`, and `selected_radar_rows`
when available.  The adapter returns CSV-friendly rows with
`combined_objective = total_score` for backward-compatible RaFT-UAV outputs.
