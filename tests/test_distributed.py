import json
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_rl.replay import Transition, save_demos


def run_training(tmp_path, demos, steps=3, updates=5, resume=None, actor_priority=False,
                 algorithm="hil_serl", wandb_mode="disabled"):
    command = [sys.executable, str(Path(__file__).with_name("viewer_task_runner.py")),
               "train", "--algorithm", algorithm, "--wandb-mode", wandb_mode, "--demos", str(demos),
               "--output", str(tmp_path / "run"), "--steps", str(steps),
               "--learner-updates", str(updates), "--batch-size", "2",
               "--capacity", "4", "--learning-starts", "1", "--random-steps", "1",
               "--sync-every", "1", "--queue-size", "1", "--save-every", "2",
               "--log-every", "1", "--max-episode-steps", "2", "--device", "cpu",
               "--control-hz", "0", "--actor-device", "cpu", "--actor-threads", "1", "--learner-threads", "1",
               "--scratch-vision"]
    if resume is not None:
        command += ["--resume", str(resume)]
    if actor_priority:
        command += ["--actor-priority"]
    return subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)


def test_auto_serl_actor_learner_integration(tmp_path):
    observation = {"state": np.zeros(18, dtype=np.float32),
                   "front": np.zeros((128, 128, 3), dtype=np.uint8),
                   "wrist": np.zeros((128, 128, 3), dtype=np.uint8)}
    demo = tmp_path / "reference.npz"
    save_demos(demo, [Transition(observation, np.array([0, 0, 0, 0], dtype=np.float32),
                                 0.0, observation, False),
                      Transition(observation, np.array([0, 0, 0, 2], dtype=np.float32),
                                 1.0, observation, True)])
    result = run_training(tmp_path, demo, algorithm="auto_serl")
    assert result.returncode == 0, result.stderr
    with np.load(tmp_path / "run" / "online.npz") as data:
        assert len(data["reward"]) == 3


@pytest.mark.parametrize("algorithm", ["hil_serl", "auto_serl"])
def test_wandb_offline_records_all_algorithms(tmp_path, algorithm):
    observation = {"state": np.zeros(18, dtype=np.float32),
                   "front": np.zeros((128, 128, 3), dtype=np.uint8),
                   "wrist": np.zeros((128, 128, 3), dtype=np.uint8)}
    demo = tmp_path / "demo.npz"
    save_demos(demo, [Transition(observation, np.array([0, 0, 0, 0], dtype=np.float32),
                                 0.0, observation, False),
                      Transition(observation, np.array([0, 0, 0, 2], dtype=np.float32),
                                 1.0, observation, True)])
    result = run_training(tmp_path, demo, algorithm=algorithm, wandb_mode="offline")
    assert result.returncode == 0, result.stderr
    assert list((tmp_path / "run" / "wandb").glob("offline-run-*/run-*.wandb"))
    logs = [json.loads(line) for line in (tmp_path / "run" / "learner_metrics.jsonl").read_text().splitlines()]
    assert any("critic" in item for item in logs)


def test_distributed_training_drains_queue_and_resumes(tmp_path):
    observation = {"state": np.zeros(18, dtype=np.float32),
                   "front": np.zeros((128, 128, 3), dtype=np.uint8),
                   "wrist": np.zeros((128, 128, 3), dtype=np.uint8)}
    demos = tmp_path / "demo.npz"
    save_demos(demos, [Transition(observation, np.array([0, 0, 0, 1], dtype=np.float32),
                                 0.0, observation, False, intervention=True)])
    result = run_training(tmp_path, demos)
    assert result.returncode == 0, result.stderr
    assert '"stage": "demos_loaded"' in result.stdout
    assert '"stage": "model_loaded"' in result.stdout
    assert '"role": "learner"' in result.stdout
    assert '"role": "actor"' in result.stdout
    process_ids = {event["role"]: event["pid"] for line in result.stdout.splitlines()
                   if (event := json.loads(line)).get("role") in ("actor", "learner")}
    assert process_ids["actor"] != process_ids["learner"]
    actor_metrics = [json.loads(line) for line in (tmp_path / "run" / "actor_metrics.jsonl").read_text().splitlines()]
    assert len(actor_metrics) == 1
    assert actor_metrics[0]["episode_steps"] == 2
    assert actor_metrics[0]["interventions"] == 1
    assert actor_metrics[0]["success_rate"] == 0.0
    assert actor_metrics[0]["policy_success_rate"] is None
    assert actor_metrics[0]["episode_duration_s"] >= 0
    assert actor_metrics[0]["steps_per_second"] > 0
    actor_timings = [json.loads(line) for line in (tmp_path / "run" / "actor_timing.jsonl").read_text().splitlines()]
    assert actor_timings[0]["window_steps"] == 3
    assert actor_timings[0]["queue_mean_ms"] >= 0
    learner_metrics = [json.loads(line) for line in (tmp_path / "run" / "learner_metrics.jsonl").read_text().splitlines()]
    assert learner_metrics[-1]["event"] == "finished"
    assert learner_metrics[-1]["updates"] == 5
    assert any(event.get("update_duration_s", 0) > 0 for event in learner_metrics)
    checkpoint = tmp_path / "run" / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["step"] == 3
    assert saved["updates"] == 5
    with np.load(tmp_path / "run" / "online.npz") as data:
        assert len(data["reward"]) == 3
    with np.load(tmp_path / "run" / "offline.npz") as data:
        assert len(data["reward"]) == 2
        assert data["intervention"][-1]
        assert data["action"][-1, 3] == 2

    resumed = run_training(tmp_path, demos, steps=5, updates=7, resume=checkpoint, actor_priority=True)
    assert resumed.returncode == 0, resumed.stderr
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["step"] == 5
    assert saved["updates"] == 7
    with np.load(tmp_path / "run" / "online.npz") as data:
        assert len(data["reward"]) == 4
    assert saved["online_replay"]["cursor"] == 1
    sessions = [json.loads(line) for line in (tmp_path / "run" / "sessions.jsonl").read_text().splitlines()]
    assert [event["event"] for event in sessions] == ["started", "finished", "started", "finished"]
    assert sessions[0]["session_id"] != sessions[2]["session_id"]
    assert sessions[0]["task"] == sessions[2]["task"] == "pick_cube"
    assert sessions[0]["algorithm"] == sessions[2]["algorithm"] == "hil_serl"
    assert sessions[1]["wall_time_s"] > 0
    assert sessions[3]["wall_time_s"] > 0
    assert sessions[2]["resume"] == str(checkpoint)
    assert {event["session_id"] for event in actor_metrics} == {sessions[0]["session_id"]}
    resumed_updates = [json.loads(line) for line in (tmp_path / "run" / "learner_metrics.jsonl").read_text().splitlines()
                       if json.loads(line)["session_id"] == sessions[2]["session_id"]
                       and "updates_per_second" in json.loads(line)]
    assert resumed_updates[0]["updates"] == 6
    assert resumed_updates[0]["updates_per_second"] <= 1 / resumed_updates[0]["update_duration_s"]


def test_learner_startup_failure_is_reported(tmp_path):
    result = run_training(tmp_path, tmp_path / "missing.npz", steps=1)
    assert result.returncode != 0
    assert "Learner failed" in result.stderr


def test_collect_records_executed_human_actions(tmp_path):
    demos = tmp_path / "collected.npz"
    command = [sys.executable, str(Path(__file__).with_name("viewer_task_runner.py")),
               "collect", "--output", str(demos), "--steps", "4"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    with np.load(demos) as data:
        assert len(data["reward"]) == 2
        assert np.all(data["intervention"])
        assert np.all(data["action"][:, 3] == 2)


def test_ctrl_c_drains_and_saves(tmp_path):
    observation = {"state": np.zeros(18, dtype=np.float32),
                   "front": np.zeros((128, 128, 3), dtype=np.uint8),
                   "wrist": np.zeros((128, 128, 3), dtype=np.uint8)}
    demos = tmp_path / "demo.npz"
    save_demos(demos, [Transition(observation, np.array([0, 0, 0, 1], dtype=np.float32),
                                 0.0, observation, False, intervention=True)])
    command = [sys.executable, str(Path(__file__).with_name("viewer_task_runner.py")),
               "train", "--wandb-mode", "disabled", "--demos", str(demos),
               "--output", str(tmp_path / "run"), "--steps", "100", "--batch-size", "2",
               "--capacity", "4", "--learning-starts", "1", "--random-steps", "1",
               "--sync-every", "1", "--queue-size", "1", "--save-every", "50",
               "--log-every", "1", "--device", "cpu", "--actor-device", "cpu",
               "--control-hz", "0",
               "--actor-threads", "1", "--learner-threads", "1", "--scratch-vision"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        deadline = time.monotonic() + 30
        observed_update = False
        while time.monotonic() < deadline and process.poll() is None:
            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if ready and '"role": "learner"' in process.stdout.readline():
                observed_update = True
                break
        assert observed_update, "Learner did not start updating"
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=45)
        assert process.returncode != 0
        saved = torch.load(tmp_path / "run" / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert saved["step"] >= 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
