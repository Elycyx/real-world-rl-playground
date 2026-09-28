"""Exercise multiprocessing with an interactive-task-shaped test double."""

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from robot_rl.cli import main
from robot_rl.registry import TASKS, Task
from robot_rl.tasks.pick_cube import actual_action, pack_observation


class InteractiveTaskDouble(Env):
    def __init__(self):
        self.action_space = Box(low=np.array([-1, -1, -1, 0], dtype=np.float32),
                                high=np.array([1, 1, 1, 2], dtype=np.float32))
        self.episode_step = 0

    def observation(self):
        return {"agent_pos": np.zeros(18, dtype=np.float32),
                "pixels": {camera: np.zeros((128, 128, 3), dtype=np.uint8)
                           for camera in ("front", "wrist")}}

    def reset(self, seed=None):
        self.episode_step = 0
        return self.observation(), {}

    def step(self, action):
        self.episode_step += 1
        override = self.episode_step == 2
        executed = np.array([0, 0, 0, 2], dtype=np.float32) if override else action
        return self.observation(), 0.0, False, override, {
            "teleop_action": executed, "is_intervention": override,
            "succeed": False, "discrete_penalty": 0.0,
        }

    def close(self):
        pass


def make_interactive_task(*, gamepad=False, seed=0, max_episode_steps=150):
    return InteractiveTaskDouble()


TASKS["pick_cube"] = Task(make_interactive_task, pack_observation, actual_action)

if __name__ == "__main__":
    main()
