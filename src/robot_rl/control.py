"""Pace interactive actions to MuJoCo's control period."""

import time


def wait_for_tick(next_tick: float, control_hz: float) -> float:
    if control_hz == 0:
        return time.perf_counter()
    remaining = next_tick - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    return max(next_tick, time.perf_counter()) + 1 / control_hz
