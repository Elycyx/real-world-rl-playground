from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from robot_rl.keyboard import KeyboardIntervention, ManipulationKeyboard


def character(name):
    return SimpleNamespace(char=name)


def special(name):
    return SimpleNamespace(name=name)


def test_keyboard_motion_gripper_and_space_edge_trigger():
    controller = ManipulationKeyboard()
    controller.press(special("space"))
    controller.press(special("space"))
    assert controller.should_intervene()
    controller.press(character("w"))
    controller.press(character("A"))
    controller.press(character("q"))
    assert controller.get_deltas() == (1, 1, 1)
    controller.release(character("w"))
    controller.release(character("a"))
    controller.release(character("q"))
    controller.press(character("d"))
    controller.press(character("e"))
    assert controller.get_deltas() == (0, -1, -1)
    controller.release(character("e"))
    controller.press(character("g"))
    controller.press(character("g"))
    assert controller.consume_gripper_toggle()
    assert not controller.consume_gripper_toggle()
    controller.release(character("g"))
    controller.press(character("g"))
    assert controller.consume_gripper_toggle()
    controller.release(special("space"))
    controller.press(special("space"))
    assert not controller.should_intervene()
    controller.release(character("g"))
    controller.press(character("g"))
    assert not controller.consume_gripper_toggle()
    controller.press(special("enter"))
    assert controller.get_episode_end_status() == "success"
    assert controller.get_episode_end_status() is None
    controller.press(special("backspace"))
    assert controller.get_episode_end_status() == "failure"
    controller.press(special("esc"))
    assert controller.get_episode_end_status() == "failure"
    controller.reset()
    assert not controller.should_intervene()
    assert controller.get_deltas() == (0, 0, 0)
    assert controller.gripper_command() == "no-op"
    assert not controller.consume_gripper_toggle()


def test_keyboard_intervention_records_physical_gripper_and_stops_listener(monkeypatch):
    started = []
    stopped = []
    monkeypatch.setattr(ManipulationKeyboard, "start", lambda self: started.append(True))
    monkeypatch.setattr(ManipulationKeyboard, "stop", lambda self: stopped.append(True))

    class Task(gym.Env):
        action_space = Box(low=np.array([-1, -1, -1, 0], dtype=np.float32),
                           high=np.array([1, 1, 1, 2], dtype=np.float32))

        def __init__(self):
            self.gripper_ctrl = 0.0

        def step(self, action):
            if action[3] == 2:
                self.gripper_ctrl = 255.0
            elif action[3] == 0:
                self.gripper_ctrl = 0.0
            return {}, 0.0, False, False, {"succeed": False}

        def get_gripper_pose(self):
            return np.array([self.gripper_ctrl], dtype=np.float32)

        def reset(self, **kwargs):
            return {}, {}

        def close(self):
            stopped.append("environment")

    env = KeyboardIntervention(Task())
    assert started == [True]
    env.reset()
    proposed = np.array([0.25, -0.5, 0, 1], dtype=np.float32)
    _, _, _, _, info = env.step(proposed)
    np.testing.assert_array_equal(info["teleop_action"], proposed)
    assert not info["is_intervention"]
    env.controller.press(special("space"))
    env.controller.press(character("w"))
    env.controller.press(character("g"))
    env.controller.release(character("g"))
    _, _, _, _, info = env.step(np.array([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(info["teleop_action"], [1, 0, 0, 2])
    assert info["is_intervention"]
    _, _, _, _, info = env.step(np.array([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(info["teleop_action"], [1, 0, 0, 1])
    env.controller.press(character("g"))
    _, _, _, _, info = env.step(np.array([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(info["teleop_action"], [1, 0, 0, 0])
    env.controller.release(character("g"))
    env.unwrapped.gripper_ctrl = 255.0
    env.controller.press(character("g"))
    _, _, _, _, info = env.step(np.array([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(info["teleop_action"], [1, 0, 0, 0])
    env.controller.press(special("enter"))
    _, reward, terminated, _, info = env.step(np.array([0, 0, 0, 1], dtype=np.float32))
    assert terminated and reward == 1.0 and info["next.success"]
    env.close()
    assert stopped == [True, "environment"]
