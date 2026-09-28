"""Trajectory-guided intervention with the repository's PickCube action format."""

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from robot_rl.algorithms.auto_serl import (
    AutoIntervention,
    AutoSerlConfig,
    load_reference,
)
from robot_rl.replay import Transition, save_demos
from robot_rl.rollout import transition


class PickCubeDouble(Env):
    def __init__(self, tcp=(0.0, 0.0, 0.0)):
        self.tcp = np.asarray(tcp, dtype=np.float32)
        self.action_space = Box(low=-1, high=2, shape=(4,), dtype=np.float32)
        self.human_action = None
        self.success = False
        self.last_action = None

    def observation(self):
        state = np.zeros(18, dtype=np.float32)
        state[-3:] = self.tcp
        return {"agent_pos": state, "pixels": {key: np.zeros((2, 2, 3), dtype=np.uint8)
                                               for key in ("front", "wrist")}}

    def reset(self, **kwargs):
        return self.observation(), {}

    def step(self, action):
        self.last_action = self.human_action if self.human_action is not None else action
        self.tcp = self.tcp + self.last_action[:3] * 0.025
        return self.observation(), float(self.success), self.success, False, {
            "teleop_action": self.last_action.copy(), "is_intervention": self.human_action is not None,
        }

    def close(self):
        pass


def reference():
    positions = np.array([[0., 0, 0], [0.025, 0, 0], [0.05, 0, 0],
                          [0.075, 0, 0]], dtype=np.float32)
    actions = np.array([[1., 0, 0, 0], [1., 0, 0, 2], [1., 0, 0, 1]], dtype=np.float32)
    return positions, actions


def test_reference_reads_one_successful_contiguous_episode(tmp_path):
    path = tmp_path / "success.npz"
    positions, actions = reference()
    items = []
    for index, action in enumerate(actions):
        before = np.zeros(18, dtype=np.float32)
        after = before.copy()
        before[-3:] = positions[index]
        after[-3:] = positions[index + 1]
        observation = {"state": before, "front": np.zeros((2, 2, 3), dtype=np.uint8),
                       "wrist": np.zeros((2, 2, 3), dtype=np.uint8)}
        next_observation = {**observation, "state": after}
        items.append(Transition(observation, action, float(index == 2), next_observation, index == 2))
    save_demos(path, items)
    recorded, commands = load_reference(path)
    np.testing.assert_array_equal(recorded, positions)
    np.testing.assert_array_equal(commands, actions)
    save_demos(path, items[:-1])
    with pytest.raises(ValueError, match="successful"):
        load_reference(path)


def test_guidance_records_executed_action_and_human_has_priority():
    raw = PickCubeDouble((0.0, -0.05, 0.0))
    env = AutoIntervention(raw, *reference())
    observation, _ = env.reset()
    proposed = np.array([0, 0, 0, 1], dtype=np.float32)
    next_observation, reward, done, _, info = env.step(proposed)
    assert info["auto_intervention"] == "guidance"
    assert info["is_intervention"]
    np.testing.assert_array_equal(info["teleop_action"], raw.last_action)
    item = transition(observation, proposed, reward, next_observation, done, info)
    np.testing.assert_array_equal(item.action, raw.last_action)
    assert item.intervention
    raw.human_action = np.array([0, 1, 0, 2], dtype=np.float32)
    _, _, _, _, info = env.step(proposed)
    assert "auto_intervention" not in info
    np.testing.assert_array_equal(info["teleop_action"], raw.human_action)
    env.close()


def test_stagnation_recovery_and_early_success_disables_guidance():
    raw = PickCubeDouble((0.025, 0, 0))
    env = AutoIntervention(raw, *reference(), AutoSerlConfig(l_stag=2, l_term=3))
    env.reset()
    action = np.array([0, 0, 0, 1], dtype=np.float32)
    env.step(action)
    _, _, _, _, info = env.step(action)
    assert info["auto_intervention"] == "recovery"
    raw.success = True
    _, _, done, _, info = env.step(action)
    assert done
    assert info["auto_intervention_disabled"]
    env.reset()
    _, _, _, _, info = env.step(action)
    assert "auto_intervention" not in info


def test_rejects_invalid_thresholds_and_indices():
    positions, actions = reference()
    with pytest.raises(ValueError, match="th1"):
        AutoIntervention(PickCubeDouble(), positions, actions, AutoSerlConfig(th1=0.02))
    with pytest.raises(ValueError, match="Recovery indices"):
        AutoIntervention(PickCubeDouble(), positions, actions, AutoSerlConfig(recover_point0=2))
