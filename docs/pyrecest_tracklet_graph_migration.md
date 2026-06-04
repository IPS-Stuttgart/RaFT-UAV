# PyRecEst tracklet graph migration

RaFT-UAV keeps Fortem-specific segment construction, quarantine flags, RF
contradiction features, and command-line options.  Generic DAG and k-best path
enumeration should use `pyrecest.tracking`.

Typical integration inside the local `global-tracklet` path:

```python
from raft_uav.baselines.pyrecest_tracklet_graph import (
    FortemTrackletGraphConfig,
    fortem_tracklet_from_summary,
    fortem_tracklet_paths_to_rows,
    rank_fortem_tracklet_paths,
)

tracklets = [fortem_tracklet_from_summary(row) for row in segment_rows]
paths = rank_fortem_tracklet_paths(
    tracklets,
    config=FortemTrackletGraphConfig(
        top_k_paths=args.global_tracklet_top_k_paths,
        beam_width=args.global_tracklet_beam_width,
        max_link_gap_s=args.global_tracklet_max_link_gap_s,
        max_transition_speed_mps=args.global_tracklet_max_transition_speed_mps,
        switch_penalty=args.global_tracklet_switch_penalty,
        coverage_reward_per_row=args.global_tracklet_coverage_reward_per_row,
        use_diverse_paths=args.global_tracklet_path_diversity_mode != "off",
        diversity_weight=args.global_tracklet_path_diversity_weight,
    ),
)
path_rows = fortem_tracklet_paths_to_rows(paths)
```

After this conversion, replay scoring should use the PyRecEst hypothesis replay
ranking adapter rather than local score arithmetic.
