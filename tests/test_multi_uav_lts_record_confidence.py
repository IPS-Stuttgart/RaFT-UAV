from __future__ import annotations

import pytest

from raft_uav.multi_uav_lts._records import parse_detection_text


def _detection_text(confidence: float) -> str:
    return f"1,1,0,0,10,10,{confidence},1,1\n"


@pytest.mark.parametrize("confidence", [-1.0, 0.0, 1.0])
def test_parse_detection_text_accepts_confidence_domain_endpoints(
    confidence: float,
) -> None:
    rows = parse_detection_text(
        _detection_text(confidence),
        source="prediction.txt",
    )

    assert len(rows) == 1
    assert rows[0].confidence == confidence


@pytest.mark.parametrize("confidence", [-1.000001, 1.000001])
def test_parse_detection_text_rejects_out_of_range_confidence(
    confidence: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"prediction\.txt:1: confidence must be in \[-1, 1\]",
    ):
        parse_detection_text(
            _detection_text(confidence),
            source="prediction.txt",
        )
