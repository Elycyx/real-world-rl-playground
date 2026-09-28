import queue
import sys
from types import SimpleNamespace

import pytest

from robot_rl.tracking import WandbTracker, wandb_values


def test_wandb_values_have_separate_actor_and_learner_axes():
    episode = wandb_values({"role": "actor", "step": 12, "session_id": "abc",
                            "episode_reward": 1.0, "policy_success_rate": None,
                            "interventions": 2, "auto_interventions": 1})
    assert episode == {"environment_step": 12, "episode/episode_reward": 1.0,
                       "episode/interventions": 2.0, "episode/auto_interventions": 1.0}
    timing = wandb_values({"role": "actor", "step": 13, "event": "timing", "policy_mean_ms": 2.5})
    assert timing == {"environment_step": 13, "timing/policy_mean_ms": 2.5}
    learner = wandb_values({"role": "learner", "step": 15, "updates": 3, "critic": 0.25})
    assert learner == {"learner_updates": 3, "learner/environment_step": 15,
                       "learner/critic": 0.25}


def test_tracker_logs_both_processes_into_one_run(monkeypatch, tmp_path):
    class FakeRun:
        def __init__(self):
            self.metrics = []
            self.records = []
            self.exit_code = None

        def define_metric(self, name, step_metric):
            self.metrics.append((name, step_metric))

        def log(self, values):
            self.records.append(values)

        def finish(self, exit_code):
            self.exit_code = exit_code

    run = FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run))
    settings = SimpleNamespace(wandb_project="test", wandb_mode="offline", session_id="test-id",
                               task="pick_cube", algorithm="hil_serl", output=tmp_path)
    events = queue.Queue()
    tracker = WandbTracker(settings, {"seed": 42}, events)
    events.put({"role": "actor", "step": 5, "episode_reward": 1})
    events.put({"role": "learner", "step": 5, "updates": 2, "critic": 0.2})
    tracker.finish(True)
    assert run.metrics == [("episode/*", "environment_step"),
                           ("timing/*", "environment_step"),
                           ("learner/*", "learner_updates")]
    assert run.records[0]["episode/episode_reward"] == 1
    assert run.records[1]["learner/critic"] == 0.2
    assert run.exit_code == 0


def test_tracker_finishes_failed_run_on_logging_error(monkeypatch, tmp_path):
    class FakeRun:
        def define_metric(self, name, step_metric):
            pass

        def log(self, values):
            raise RuntimeError("logging unavailable")

        def finish(self, exit_code):
            assert exit_code == 1

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: FakeRun()))
    settings = SimpleNamespace(wandb_project="test", wandb_mode="offline", session_id="test-id",
                               task="pick_cube", algorithm="hil_serl", output=tmp_path)
    events = queue.Queue()
    tracker = WandbTracker(settings, {}, events)
    events.put({"role": "actor", "step": 1, "episode_reward": 1})
    with pytest.raises(RuntimeError, match="W&B failed"):
        tracker.finish(True)
