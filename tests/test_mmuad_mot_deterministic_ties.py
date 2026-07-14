from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from raft_uav.mmuad.mot import MultiObjectTrackerConfig, _nearest_track_id


class _ReverseIterationSet(set[int]):
    """Expose an iteration order opposite to the desired MOT tie break."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self), reverse=True))


def test_nearest_track_tie_uses_lowest_track_id() -> None:
    active = {
        1: SimpleNamespace(state=np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        2: SimpleNamespace(state=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    }
    unmatched_tracks = _ReverseIterationSet({1, 2})

    assert list(unmatched_tracks) == [2, 1]
    selected = _nearest_track_id(
        np.array([0.0, 0.0, 0.0]),
        active,
        unmatched_tracks,
        MultiObjectTrackerConfig(max_association_distance_m=2.0),
    )

    assert selected == 1
