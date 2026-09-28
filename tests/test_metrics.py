import pytest

from robot_rl.metrics import EpisodeMetrics, StepTimings


def test_step_timings_summarizes_and_resets():
    timings = StepTimings()
    timings.add(sync=0.01, policy=0.02, environment=0.03, queue=0.04)
    timings.add(sync=0.03, policy=0.04, environment=0.05, queue=0.06)
    result = timings.flush(10)
    assert result["step"] == 10
    assert result["window_steps"] == 2
    assert result["policy_mean_ms"] == 30
    assert result["queue_max_ms"] == 60
    assert timings.count == 0


def test_episode_metrics_separates_policy_success_from_interventions_and_warmup(monkeypatch):
    now = iter(range(10))
    monkeypatch.setattr("robot_rl.metrics.time.perf_counter", lambda: next(now))
    metrics = EpisodeMetrics(10)
    metrics.add(0.0, False, True)
    warmup = metrics.finish(11, True, False)
    assert warmup["policy_success_rate"] is None
    assert warmup["success_rate"] == 1.0
    metrics.reset(11)
    metrics.add(1.0, True, False)
    assisted = metrics.finish(12, True, False)
    assert assisted["policy_success_rate"] is None
    assert assisted["success_rate"] == 1.0
    metrics.reset(12)
    metrics.add(0.0, False, False)
    policy = metrics.finish(13, False, True)
    assert policy["policy_success_rate"] == 0.0
    assert policy["policy_episodes"] == 1
    assert policy["success_rate"] == pytest.approx(2 / 3)
    assert policy["success_rate_last_20"] == pytest.approx(2 / 3)
    assert policy["episode_steps"] == 1
    assert policy["episode_duration_s"] == 1
    assert policy["truncated"]


def test_episode_metrics_separates_auto_recovery_from_human_interventions():
    metrics = EpisodeMetrics(0)
    metrics.add(0, True, False, "guidance")
    metrics.add(0, True, False, "recovery")
    metrics.add(1, True, False)
    result = metrics.finish(3, True, False)
    assert result["interventions"] == 3
    assert result["auto_interventions"] == 2
    assert result["human_interventions"] == 1
    assert result["recovery_steps"] == 1
    metrics.reset(3)
    assert metrics.finish(4, False, False)["auto_interventions"] == 0
