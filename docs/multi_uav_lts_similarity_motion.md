# Reliability-gated similarity common motion for Multi-UAV LTS

## Hypothesis

The maintained proposal graph subtracts one robust shared translation from every
candidate transition. That is intentionally safe, but it cannot explain camera
zoom, small in-plane rotation, or a coherent expanding/contracting swarm. Those
effects can make a correct continuation appear to accelerate and can make two
crossing assignments nearly indistinguishable.

This experiment extends the common-motion step from translation to a two-
dimensional similarity transform,

\[
p_{t+1}=s_t R(\theta_t)p_t+d_t,
\]

while retaining the maintained translation estimate as the exact fallback.

## Safety contract

Similarity motion is opt-in. The default remains `translation`, and no existing
candidate changes unless `--common-motion-model similarity` is supplied together
with `--enable-common-motion`.

For every consecutive frame pair, the implementation:

1. obtains the maintained robust translation estimate when available;
2. forms reciprocal proposal correspondences after that initial alignment;
3. fits an isotropic scale, rotation, and translation with robust refinement;
4. rejects the richer model unless it has enough support and spatial spread;
5. rejects excessive scale or rotation;
6. requires a bounded normalized residual; and
7. requires a material residual improvement over translation.

A rejected, ill-conditioned, or implausible fit uses the exact maintained
translation step. If neither translation nor similarity is reliable, the frame
has no common-motion step.

The similarity transform is composed across gaps. Box centers, axis-aligned box
envelopes, residual velocities, delayed continuation scores, and ambiguity-beam
acceleration diagnostics all use the same transform semantics.

## Direct experiment

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.experimental_proposal_graph_tracker \
  /path/to/proposals \
  --first-frame-label-dir /path/to/TrainLabels_FirstFrameOnly \
  --output-dir /path/to/similarity/predictions \
  --output-json /path/to/similarity/summary.json \
  --enable-common-motion \
  --common-motion-model similarity \
  --common-motion-min-pairs 4 \
  --common-motion-max-normalized-step 8 \
  --common-motion-max-normalized-residual 1.5 \
  --similarity-min-pairs 4 \
  --similarity-max-scale-change 0.12 \
  --similarity-max-rotation-deg 10 \
  --similarity-max-normalized-residual 1 \
  --similarity-min-normalized-spread 2 \
  --similarity-min-residual-improvement 0.05
```

The higher-value candidate combines the model with delayed atomic path cover and
the ambiguity-triggered beam:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.experimental_proposal_graph_tracker \
  /path/to/proposals \
  --first-frame-label-dir /path/to/TestLabels_FirstFrameOnly \
  --output-dir /path/to/delayed_similarity/predictions \
  --enable-delayed-path-cover \
  --enable-ambiguity-beam \
  --enable-common-motion \
  --common-motion-model similarity
```

Run both fixed candidates through the complete native-resolution evidence suite:

```bash
python scripts/run_multi_uav_lts_similarity_evidence.py \
  --inputs-json /path/to/public-inputs.json \
  --run-dir /path/to/run \
  --device 0 \
  --expected-sequence-count 102 \
  --expected-frame-count 77293 \
  --require-improvement
```

## Evaluation

Compare the complete out-of-fold similarity candidate against the identical raw
seeded tracker and the translation-common-motion candidate with the existing
guarded tournament. Acceptance still requires:

- positive held-out `CODABENCH_HOTA` gain;
- a positive paired-bootstrap lower bound;
- acceptable MOTA and IDF1 changes;
- no excessive worst-prefix regression;
- complete sequence coverage; and
- exact source and configuration provenance.

The expected successful signature is higher AssA/IDF1 and fewer identity
switches on sequences with coherent zoom, rotation, or swarm expansion. A gain
caused only by additional output rows is not evidence for the common-motion
hypothesis.

No hidden-test or leaderboard improvement is claimed until the complete
102-sequence training-side tournament selects the transformed candidate over
`raw`.
