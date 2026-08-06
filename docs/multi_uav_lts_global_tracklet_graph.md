# Multi-UAV LTS global tracklet-graph experiment

This experiment tests the result-oriented hypothesis suggested by the permissive
proposal-bank audit: detector proposals usually contain the UAV, while online
proposal pruning, births, deaths, and identity assignment leave substantial
headroom.

The implementation is **non-oracle**. It consumes only:

- low-threshold detector proposals;
- the first-frame labels supplied by the LTS protocol; and
- fixed configuration values selected before hidden-test inference.

Training truth is used only by the existing evaluator and guarded tournament.
The identity-oracle outputs produced by the proposal audit must never be passed
to this tracker or packaged as a submission.

## Method

`raft_uav.multi_uav_lts.global_tracklet_graph` performs four stages.

1. It confidence-filters and deterministically suppresses near-duplicate
   proposals. Exact first-frame labels replace overlapping detector proposals,
   preserving the benchmark identities without depending on detector row order.
2. It forms short tracklets from reciprocal, margin-separated consecutive-frame
   assignments. Ambiguous crossings are split instead of being committed
   immediately.
3. It builds sparse candidate edges between tracklets and solves one global
   maximum-gain bipartite matching. Link costs combine predicted center error,
   size change, velocity disagreement, overlap, gap length, and proposal
   confidence. Every tracklet has at most one predecessor and successor.
4. Paths connected to first-frame labels retain those IDs. Persistent unseeded
   paths can become confirmed late births. Short unseeded clutter is removed.
   An optional same-border re-entry rule permits longer gaps without enabling
   unrestricted long-distance links.

The global assignment is implemented with SciPy's sparse full bipartite matcher.
Only the best bounded number of successor edges per tracklet is materialized, so
memory scales with candidate links rather than the square of the proposal count.

## Focused synthetic regressions

Run:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_multi_uav_lts_global_tracklet_graph.py
```

The suite covers:

- persistent late-birth confirmation;
- strict no-birth control;
- single-frame clutter rejection;
- delayed identity recovery across a crossing gap;
- same-border exit and re-entry;
- one-to-one global links;
- reciprocal rejection of an exact local-assignment tie;
- proposal-row-order invariance;
- sparse total-gain optimality on a controlled graph; and
- destructive-path and malformed-configuration guards.

These tests establish algorithmic behavior. They do not establish a competition
improvement.

## One-sequence command

```bash
PYTHONPATH=src python -m raft_uav.multi_uav_lts.global_tracklet_graph \
  /path/to/low_threshold_proposals \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --output-dir outputs/global_tracklet_graph_births \
  --min-confidence 0.003 \
  --global-max-gap 12 \
  --min-birth-frames 3 \
  --min-birth-span 2 \
  --min-birth-mean-confidence 0.01
```

For the border-re-entry candidate, add explicit original image dimensions:

```bash
  --enable-border-reentry \
  --border-max-gap 90 \
  --frame-width 1920 \
  --frame-height 1080
```

Verify the dimensions against the extracted sequence files before relying on
that candidate. The ordinary and no-birth candidates do not use image bounds.

Each run writes prediction `.txt` files plus:

```text
global_tracklet_graph_summary.json
global_tracklet_graph_sequences.csv
global_tracklet_graph_links.csv
```

The link table records costs, gains, gaps, and whether the special border rule
was used.

## Complete train-split experiment

Dispatch the GitHub Actions workflow:

```text
Multi-UAV LTS global tracklet graph
```

It runs on `[self-hosted, Linux, X64, nvidia-smi]`. Start with
`probe_only=true`; the probe requires exact agreement between the 102 truth,
first-frame-label, and proposal manifests.

A normal run generates four fixed ablations:

| Candidate | Late births | Border re-entry | Local assignment |
|---|---:|---:|---|
| `graph_no_births` | no | no | reciprocal |
| `graph_births` | yes | no | reciprocal |
| `graph_births_greedy_local` | yes | no | ordinary Hungarian |
| `graph_births_border` | yes | yes | reciprocal |

The workflow then invokes the existing guarded tournament with the unmodified
seeded BoT-SORT predictions as the raw control. A transformed candidate is
eligible only when it satisfies the configured mean HOTA gain, paired-bootstrap
confidence bound, MOTA and IDF1 floors, scenario-prefix robustness, and complete
sequence coverage.

## Interpreting results

The ablations answer separate questions:

- `graph_no_births` versus `graph_births` measures whether controlled late births
  overcome the measured no-birth recall ceiling.
- `graph_births` versus `graph_births_greedy_local` measures whether refusing
  ambiguous local links helps the later global assignment.
- `graph_births` versus `graph_births_border` isolates the value of explicit
  partial-out-of-view re-entry handling.
- Any graph candidate versus raw measures whether proposal-level offline
  association recovers enough detections and identities to compensate for extra
  false positives or localization errors.

Canonical `DetA`, `AssA`, and `LocA` should be inspected alongside the exported
`CODABENCH_HOTA`, `CODABENCH_MOTA`, and `CODABENCH_IDF1` fields. A DetA gain with
an AssA loss suggests association costs need work; an AssA gain with a MOTA loss
suggests births or long-gap links are too permissive.

Do not prepare a test submission unless the guarded train-split result selects a
non-raw candidate. A raw fallback is the expected result when the hypothesis is
not supported.
