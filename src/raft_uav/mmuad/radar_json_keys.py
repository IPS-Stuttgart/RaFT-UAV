from __future__ import annotations

RADAR_NESTED_TABLE_KEYS = tuple(
    (
        "radar_polar radar_detections detections tracks targets objects "
        "measurements returns rows data"
    ).split()
)
RADAR_SEQUENCE_KEYS = tuple("sequence_id sequence seq scene scene_id".split())
RADAR_TIME_KEYS = tuple(
    (
        "time_s timestamp timestamp_s stamp_s time t sec secs seconds stamp stamp.sec "
        "header.stamp.sec timestamp_ns time_ns stamp_ns nanoseconds timestamp_us time_us "
        "stamp_us timestamp_usec time_usec stamp_usec microseconds timestamp_ms time_ms "
        "stamp_ms milliseconds nanosec nsec nsecs"
    ).split()
)
RADAR_PARENT_DEFAULT_KEYS = RADAR_SEQUENCE_KEYS + RADAR_TIME_KEYS
RADAR_HINT_KEYS = set(RADAR_TIME_KEYS) | set(
    (
        "range_m range r rho distance_m azimuth_rad az_rad azimuth_deg az_deg azimuth az "
        "bearing_rad bearing bearing_deg elevation_rad el_rad elevation_deg el_deg elevation "
        "el pitch_rad pitch pitch_deg"
    ).split()
)
