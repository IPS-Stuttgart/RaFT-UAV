# MMUAD constant-velocity bridge candidate quota

The branch/source reservoir protects candidate provenance, and the temporal-support
quota protects candidates with nearby observations on one or both adjacent sides.
A remaining failure mode is a false temporal chain: unrelated earlier and later
candidates can each be close to a middle-frame candidate while implying a sharp
zig-zag.

This experiment adds a stricter, truth-free recall guard for interior frames. For
each candidate, it considers candidates at the nearest earlier and later times,
forms bracketing endpoint pairs, and linearly interpolates each pair to the middle
time. A candidate receives bridge support only when:

- both side gaps are within the configured time gate;
- the previous-to-current, current-to-next, and endpoint-to-endpoint speeds pass
  the configured speed gate; and
- the interpolated position is within the configured 3D error gate.

The strongest supported candidates are added to the normal branch/source
reservoir before its final per-frame cap. No truth is used for bridge scoring or
selection. Optional truth inputs are used only for oracle-recall diagnostics.

## Example

```bash
python scripts/mmuad_constant_velocity_bridge_quota.py \
  --candidate-csv raw=outputs/mmuad_raw_candidates.csv \
  --candidate-csv translated=outputs/mmuad_translated_candidates.csv \
  --candidate-csv dynamic=outputs/mmuad_dynamic_candidates.csv \
  --output-csv outputs/mmuad_cv_bridge/candidates.csv \
  --summary-json outputs/mmuad_cv_bridge/summary.json \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --max-candidates-per-frame 40 \
  --bridge-top-n 2 \
  --max-frame-gap-s 1.0 \
  --max-speed-mps 60 \
  --max-interpolation-error-m 5 \
  --interpolation-scale-m 5
```

For train/public-validation diagnostics, add:

```bash
  --truth-csv path/to/truth.csv \
  --oracle-frame-csv outputs/mmuad_cv_bridge/oracle_frames.csv \
  --oracle-summary-csv outputs/mmuad_cv_bridge/oracle_summary.csv \
  --oracle-by-sequence-csv outputs/mmuad_cv_bridge/oracle_by_sequence.csv
```

## Output diagnostics

The selected candidate table includes:

- `candidate_cv_bridge_supported`;
- interpolation error and exponential bridge score;
- previous/current, current/next, and endpoint segment speeds;
- previous and next time gaps; and
- `cv_bridge` reservoir provenance for quota-selected rows.

`max_neighbors_per_side` bounds the endpoint-pair search by retaining the nearest
candidates to the middle hypothesis on each side. Set it to `0` for an unbounded
search when candidate counts are already small.

## Suggested train-CV grid

```text
bridge_top_n:                  0, 1, 2, 3
max_frame_gap_s:               0.25, 0.5, 1.0
max_speed_mps:                 20, 40, 60, 100
max_interpolation_error_m:     1, 2, 5, 10
interpolation_scale_m:         1, 2, 5, 10
require_same_source/branch:    off, on
```

Evaluate full/top-3/top-5/top-10 oracle recall and frozen learned-sigma Huber
mixture-MAP MSE. The intended gain is better low-K candidate recall without
promoting one-sided or zig-zag temporal coincidences. Because the test requires
bracketing frames, it complements rather than replaces one-sided temporal
support at sequence boundaries and across missing frames.
