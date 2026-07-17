from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_kalman_validation_survives_optional_runtime_hook_opt_out() -> None:
    script = textwrap.dedent(
        """
        import numpy as np

        from raft_uav.baselines.kalman import (
            AsyncConstantVelocityKalmanTracker,
            TrackingMeasurement,
        )


        def expect_value_error(action, message):
            try:
                action()
            except ValueError as exc:
                if message not in str(exc):
                    raise AssertionError(str(exc)) from exc
            else:
                raise AssertionError(f"missing ValueError: {message}")


        expect_value_error(
            lambda: TrackingMeasurement(np.nan, np.zeros(3), np.eye(3), "radar"),
            "measurement time_s must be a finite numeric timestamp",
        )
        expect_value_error(
            lambda: AsyncConstantVelocityKalmanTracker(
                np.zeros(3),
                0.0,
                initial_position_std_m=True,
            ),
            "initial_position_std_m must be a finite nonnegative scalar",
        )
        tracker = AsyncConstantVelocityKalmanTracker(np.zeros(3), 0.0)
        expect_value_error(
            lambda: tracker.predict_to(np.inf),
            "time_s must be a finite numeric timestamp",
        )
        """
    )
    env = os.environ.copy()
    env["RAFT_UAV_SKIP_RUNTIME_HOOKS"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
