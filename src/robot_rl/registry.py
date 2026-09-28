"""Explicit task and algorithm integration points for future experiments."""

from collections.abc import Callable
from dataclasses import dataclass

from robot_rl.algorithms.hil_serl import ActorReplica, HILSerl
from robot_rl.planners.pick_cube import PickCubePlanner
from robot_rl.tasks.pick_cube import actual_action, make_env, pack_observation


@dataclass(frozen=True)
class Task:
    make_env: Callable
    pack_observation: Callable
    actual_action: Callable


TASKS = {"pick_cube": Task(make_env, pack_observation, actual_action)}
ALGORITHMS = {"hil_serl": HILSerl, "auto_serl": HILSerl}
ACTORS = {"hil_serl": ActorReplica, "auto_serl": ActorReplica}
PLANNERS = {"pick_cube": PickCubePlanner}
