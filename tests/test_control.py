from robot_rl.control import wait_for_tick


def test_pacing_waits_for_tick_and_does_not_catch_up(monkeypatch):
    current = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        current[0] += seconds

    monkeypatch.setattr("robot_rl.control.time.perf_counter", lambda: current[0])
    monkeypatch.setattr("robot_rl.control.time.sleep", sleep)
    next_tick = wait_for_tick(0.0, 10.0)
    assert next_tick == 0.1
    next_tick = wait_for_tick(next_tick, 10.0)
    assert waits == [0.1]
    assert next_tick == 0.2
    current[0] = 1.0
    assert wait_for_tick(next_tick, 10.0) == 1.1
    assert wait_for_tick(1.1, 0) == 1.0
