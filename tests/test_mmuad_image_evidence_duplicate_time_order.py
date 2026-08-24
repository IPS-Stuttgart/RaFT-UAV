from pathlib import Path

from raft_uav.mmuad import image_evidence


def test_duplicate_image_timestamps_keep_discovery_order_across_package_boundary() -> None:
    image_files = [
        Path("camera_000/2.0.png"),
        Path("camera_001/2.0.png"),
        Path("camera_002/0.0.png"),
        Path("camera_003/2.0.png"),
        Path("camera_004/1.0.png"),
    ]

    public_rows = image_evidence._image_file_rows(image_files)
    implementation_rows = image_evidence._IMPL._image_file_rows(image_files)
    expected = [str(image_files[0]), str(image_files[1]), str(image_files[3])]

    assert public_rows.loc[public_rows["image_time_s"] == 2.0, "image_path"].tolist() == expected
    assert (
        implementation_rows.loc[
            implementation_rows["image_time_s"] == 2.0,
            "image_path",
        ].tolist()
        == expected
    )
