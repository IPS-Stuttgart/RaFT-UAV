from raft_uav.mmuad.submission import parse_official_classification_cell


def test_track5_public_parser_rejects_bad_class_id():
    try:
        parse_official_classification_cell(4)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
