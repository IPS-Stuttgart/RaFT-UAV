from __future__ import annotations

import pytest

from raft_uav.mmuad import submission


def test_official_classification_domain_error_mentions_allowed_ids() -> None:
    with pytest.raises(ValueError, match=r"must be one of \{0, 1, 2, 3\}; got 4"):
        submission.parse_official_classification_cell(4)
