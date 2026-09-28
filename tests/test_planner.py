from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.spaces import Box

from robot_rl.cli import collect
from robot_rl.planners.pick_cube import PickCubePlanner
from robot_rl.registry import TASKS, Task
from robot_rl.tasks.pick_cube import actual_action, pack_observation


class SimulatedPickCube:
    def __init__(self, max_episode_steps=150):
        self.max_episode_steps = max_episode_steps
        self.action_space = Box(low=np.array([-1, -1, -1, 0], dtype=np.float32),
                                high=np.array([1, 1, 1, 2], dtype=np.float32))
        self.unwrapped = self
        self.reset()

    def reset(self, seed=None):
        self.tcp = np.array([0.3, 0.0, 0.05], dtype=np.float64)
        self.block = np.array([0.5, 0.0, 0.02], dtype=np.float64)
        self.data = SimpleNamespace(mocap_pos=self.tcp[None, :].copy(),
                                    sensor=lambda name: SimpleNamespace(data=self.block))
        self.grasped = False
        self.open = False
        self.episode_steps = 0
        return self.observation(), {}

    def observation(self):
        state = np.zeros(18, dtype=np.float32)
        state[-3:] = self.tcp
        return {"agent_pos": state, "pixels": {
            camera: np.zeros((8, 8, 3), dtype=np.uint8) for camera in ("front", "wrist")}}

    def step(self, action):
        self.episode_steps += 1
        self.data.mocap_pos[0] = np.clip(self.data.mocap_pos[0] + action[:3] * 0.025,
                                         [0.2, -0.3, 0], [0.6, 0.3, 0.5])
        self.tcp += 0.7 * (self.data.mocap_pos[0] - self.tcp)
        if action[3] == 0:
            self.open = True
        if action[3] == 2:
            if self.open and np.linalg.norm(self.tcp - self.block) < 0.05:
                self.grasped = True
            self.open = False
        if self.grasped:
            self.block[:] = self.tcp
        success = self.block[2] > 0.12
        return (self.observation(), float(success), bool(success),
                self.episode_steps >= self.max_episode_steps, {"succeed": bool(success)})

    def close(self):
        pass


def test_planner_routes_above_table_then_grasps_and_lifts():
    env = SimulatedPickCube()
    observation, _ = env.reset()
    planner = PickCubePlanner()
    planner.reset(env, observation)
    safe_z = planner.waypoints[1].position[2]
    for _ in range(150):
        action = planner.act(env, observation)
        assert env.action_space.contains(action)
        if np.linalg.norm(action[:2]) > 1e-3:
            assert env.tcp[2] >= safe_z - planner.tolerance
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert reward == 1.0
    assert terminated and not truncated
    assert env.grasped


def test_collect_motion_planning_saves_complete_successful_episodes(tmp_path, monkeypatch):
    monkeypatch.setitem(TASKS, "pick_cube", Task(
        lambda **kwargs: SimulatedPickCube(kwargs["max_episode_steps"]), pack_observation, actual_action))
    destination = tmp_path / "planned.npz"
    arguments = Namespace(task="pick_cube", source="motion-planning", episodes=2, output=destination,
                          steps=300, seed=0, gamepad=False, max_episode_steps=150, control_hz=0)
    collect(arguments)
    with np.load(destination, allow_pickle=False) as data:
        assert len(data["reward"]) > 2
        assert np.count_nonzero(data["done"]) == 2
        assert np.count_nonzero(data["reward"]) == 2
        assert not np.any(data["intervention"])
        assert data["action"].shape[1] == 4


def test_collect_motion_planning_rejects_incomplete_episodes(tmp_path, monkeypatch):
    monkeypatch.setitem(TASKS, "pick_cube", Task(
        lambda **kwargs: SimulatedPickCube(kwargs["max_episode_steps"]), pack_observation, actual_action))
    destination = tmp_path / "planned.npz"
    arguments = Namespace(task="pick_cube", source="motion-planning", episodes=1, output=destination,
                          steps=2, seed=0, gamepad=False, max_episode_steps=150, control_hz=0)
    with pytest.raises(RuntimeError, match="No successful planned episodes"):
        collect(arguments)
    assert not destination.exists()


def test_collect_motion_planning_records_executed_human_override(tmp_path, monkeypatch):
    override = np.array([0.5, -0.5, 0.0, 0.0], dtype=np.float32)

    class OverriddenPickCube(SimulatedPickCube):
        def step(self, action):
            if self.episode_steps == 0:
                observation, reward, terminated, truncated, info = super().step(override)
                info.update(is_intervention=True, teleop_action=override)
                return observation, reward, terminated, truncated, info
            return super().step(action)

    monkeypatch.setitem(TASKS, "pick_cube", Task(
        lambda **kwargs: OverriddenPickCube(kwargs["max_episode_steps"]), pack_observation, actual_action))
    destination = tmp_path / "planned_override.npz"
    arguments = Namespace(task="pick_cube", source="motion-planning", episodes=1, output=destination,
                          steps=150, seed=0, gamepad=False, max_episode_steps=150, control_hz=0)
    collect(arguments)
    with np.load(destination, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["action"][0], override)
        assert data["intervention"][0]
        assert np.count_nonzero(data["done"]) == 1


def test_collect_closes_environment_when_initial_reset_fails(tmp_path, monkeypatch):
    class FailingReset(SimulatedPickCube):
        def reset(self, seed=None):
            if hasattr(self, "action_space") and getattr(self, "initialized", False):
                raise RuntimeError("reset failed")
            self.initialized = True
            return super().reset(seed=seed)

        def close(self):
            closed.append(True)

    closed = []
    monkeypatch.setitem(TASKS, "pick_cube", Task(
        lambda **kwargs: FailingReset(kwargs["max_episode_steps"]), pack_observation, actual_action))
    arguments = Namespace(task="pick_cube", source="motion-planning", episodes=1,
                          output=tmp_path / "failed.npz", steps=150, seed=0,
                          gamepad=False, max_episode_steps=150, control_hz=0)
    with pytest.raises(RuntimeError, match="reset failed"):
        collect(arguments)
    assert closed == [True]
