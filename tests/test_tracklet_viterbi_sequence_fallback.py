from __future__ import annotations

from raft_uav.baselines import tracklet_viterbi


def _node(
    frame_index: int,
    candidate_index: int | None,
    cost: float,
    payload: str,
    *,
    miss: bool = False,
) -> tracklet_viterbi.SequenceAssociationNode:
    return tracklet_viterbi.SequenceAssociationNode(
        frame_index=frame_index,
        candidate_index=candidate_index,
        unary_cost=cost,
        is_missed_detection=miss,
        payload=payload,
    )


def test_local_sequence_association_fallback_uses_miss_streak_context() -> None:
    frames = [
        [
            _node(0, None, 0.0, "miss", miss=True),
            _node(0, 0, 2.0, "candidate"),
        ],
        [
            _node(1, 0, 0.0, "reacquired"),
            _node(1, None, 3.0, "second_miss", miss=True),
        ],
    ]
    seen_miss_streaks: list[int] = []

    def transition_cost(
        previous: tracklet_viterbi.SequenceAssociationNode,
        current: tracklet_viterbi.SequenceAssociationNode,
        context: tracklet_viterbi.SequenceTransitionContext,
    ) -> float:
        del previous
        seen_miss_streaks.append(int(context.previous_miss_streak))
        if current.payload == "reacquired":
            return float(context.previous_miss_streak)
        return 0.0

    paths = tracklet_viterbi._fallback_solve_top_k_viterbi_sequence_associations(
        frames,
        transition_cost,
        top_k_terminal_paths=2,
    )

    assert len(paths) == 2
    assert [node.payload for node in paths[0].nodes] == ["miss", "reacquired"]
    assert paths[0].total_cost == 1.0
    assert paths[1].total_cost >= paths[0].total_cost
    assert {0, 1}.issubset(set(seen_miss_streaks))
