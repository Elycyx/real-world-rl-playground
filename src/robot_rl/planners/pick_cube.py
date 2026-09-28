"""Feedback-controlled Cartesian waypoint planner for MuJoCo PickCube."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Waypoint:
    position: np.ndarray
    gripper: int


class PickCubePlanner:
    def __init__(self, clearance: float = 0.12, grasp_offset: float = 0.005,
                 lift_height: float = 0.18, tolerance: float = 0.015,
                 grasp_steps: int = 8, step_size: float = 0.025):
        self.clearance = clearance
        self.grasp_offset = grasp_offset
        self.lift_height = lift_height
        self.tolerance = tolerance
        self.grasp_steps = grasp_steps
        self.step_size = step_size
        self.waypoints: list[Waypoint] = []
        self.block_position = np.zeros(3, dtype=np.float64)
        self.phase = 0
        self.grasp_count = 0

    def reset(self, env, observation: dict) -> None:
        self.block_position = np.asarray(env.unwrapped.data.sensor("block_pos").data, dtype=np.float64).copy()
        tcp_position = np.asarray(observation["agent_pos"][-3:], dtype=np.float64)
        safe_z = max(float(tcp_position[2]), float(self.block_position[2] + self.clearance))
        cube_xy = self.block_position[:2]
        self.waypoints = [
            Waypoint(np.array([*tcp_position[:2], safe_z]), 0),
            Waypoint(np.array([*cube_xy, safe_z]), 0),
            Waypoint(np.array([*cube_xy, self.block_position[2] + self.grasp_offset]), 0),
            Waypoint(np.array([*cube_xy, self.block_position[2] + self.grasp_offset]), 2),
            Waypoint(np.array([*cube_xy, self.block_position[2] + self.lift_height]), 1),
        ]
        self.phase = 0
        self.grasp_count = 0

    def act(self, env, observation: dict) -> np.ndarray:
        if not self.waypoints:
            self.reset(env, observation)
        current_block = np.asarray(env.unwrapped.data.sensor("block_pos").data, dtype=np.float64)
        if self.phase < 3 and np.linalg.norm(current_block[:2] - self.block_position[:2]) > 0.025:
            self.reset(env, observation)
        tcp_position = np.asarray(observation["agent_pos"][-3:], dtype=np.float64)
        mocap_position = np.asarray(env.unwrapped.data.mocap_pos[0], dtype=np.float64)
        if self.phase < 3:
            waypoint = self.waypoints[self.phase]
            if (np.linalg.norm(tcp_position - waypoint.position) <= self.tolerance
                    and np.linalg.norm(mocap_position - waypoint.position) <= self.tolerance):
                self.phase += 1
        elif self.phase == 3:
            self.grasp_count += 1
            if self.grasp_count > self.grasp_steps:
                self.phase = 4
        waypoint = self.waypoints[min(self.phase, 4)]
        delta = np.clip((waypoint.position - mocap_position) / self.step_size, -1, 1)
        return np.array([*delta, waypoint.gripper], dtype=np.float32)
