"""MuJoCo Panda PickCube, using gym_hil's real task and action wrappers."""

import gym_hil  # noqa: F401
import gymnasium as gym
import numpy as np

from robot_rl.keyboard import KeyboardIntervention
from robot_rl.safe_viewer import KeyboardSafeViewer


def make_env(*, gamepad: bool = False, seed: int = 0,
             max_episode_steps: int = 500, random_block_position: bool = True) -> gym.Env:
    task = "gym_hil/PandaPickCubeGamepad-v0" if gamepad else "gym_hil/PandaPickCubeKeyboard-v0"
    env = gym.make(task, image_obs=True, reward_type="sparse", random_block_position=random_block_position,
                   use_inputs_control=gamepad, use_gamepad=gamepad, use_viewer=gamepad, auto_reset=False,
                   reset_delay_seconds=0, max_episode_steps=max_episode_steps)
    if not gamepad:
        try:
            env = KeyboardSafeViewer(env)
            env = KeyboardIntervention(env)
        except Exception:
            env.close()
            raise
    return env


def pack_observation(observation: dict) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray(observation["agent_pos"], dtype=np.float32).copy(),
        "front": np.asarray(observation["pixels"]["front"], dtype=np.uint8).copy(),
        "wrist": np.asarray(observation["pixels"]["wrist"], dtype=np.uint8).copy(),
    }


def actual_action(proposed: np.ndarray, info: dict) -> np.ndarray:
    return np.asarray(info.get("teleop_action", proposed), dtype=np.float32).copy()
