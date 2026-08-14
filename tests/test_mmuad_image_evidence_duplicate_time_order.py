from pathlib import Path

from raft_uav.mmuad.image_evidence import _image_file_rows


def test_duplicate_image_timestamps_preserve_discovery_order() -> None:
    image_files = [
        Path("camera_000/2.0.png"),
        Path("camera_001/2.0.png"),
        Path("camera_002/0.0.png"),
        Path("camera_003/2.0.png"),
        Path("camera_004/1.0.png"),
    ]

    rows = _image_file_rows(image_files)

    assert rows.loc[rows["image_time_s"] == 2.0, "image_path"].tolist() == [
        str(image_files[0]),
        str(image_files[1]),
        str(image_files[3]),
    ]
