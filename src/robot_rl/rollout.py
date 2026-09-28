"""Translate environment observations and executed human actions to replay transitions."""

import numpy as np

from robot_rl.registry import TASKS
from robot_rl.replay import Transition


def transition(observation: dict, action: np.ndarray, reward: float, next_observation: dict,
               terminated: bool, info: dict, task=TASKS["pick_cube"]) -> Transition:
    return Transition(task.pack_observation(observation), task.actual_action(action, info), float(reward),
                      task.pack_observation(next_observation), bool(terminated),
                      float(info.get("discrete_penalty", 0.0)), bool(info.get("is_intervention", False)))
