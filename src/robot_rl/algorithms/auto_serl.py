"""AutoSERL-style trajectory-guided intervention for the PickCube action space."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class AutoSerlConfig:
    th1: float = 0.005
    th2: float = 0.02
    l_term: int = 10
    l_stag: int = 20
    recover_point0: int = 0
    recover_point1: int | None = None
    step_size: float = 0.025


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one successful episode from the repository's demo archive."""
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["obs_state"], dtype=np.float32)
        next_states = np.asarray(data["next_obs_state"], dtype=np.float32)
        actions = np.asarray(data["action"], dtype=np.float32)
        done = np.asarray(data["done"], dtype=bool)
        reward = np.asarray(data["reward"], dtype=np.float32)
    if (states.ndim != 2 or states.shape[1] != 18 or next_states.shape != states.shape
            or actions.shape != (len(states), 4) or done.shape != (len(states),)
            or reward.shape != (len(states),) or len(states) < 2
            or np.any(done[:-1]) or not done[-1] or reward[-1] <= 0
            or not np.isfinite(states).all() or not np.isfinite(next_states).all()
            or not np.isfinite(actions).all()):
        raise ValueError("AutoSERL reference must be one successful, contiguous PickCube episode")
    positions = np.concatenate((states[:, -3:], next_states[-1:, -3:]))
    if not np.allclose(states[1:, -3:], next_states[:-1, -3:], atol=1e-4):
        raise ValueError("AutoSERL reference transitions must form a continuous TCP trajectory")
    return positions, actions


class AutoIntervention(gym.Wrapper):
    """Sliding-window guidance, stagnation recovery, and early-success termination."""

    def __init__(self, env: gym.Env, positions: np.ndarray, actions: np.ndarray,
                 config: AutoSerlConfig | None = None):
        super().__init__(env)
        if config is None:
            config = AutoSerlConfig()
        self.positions = np.asarray(positions, dtype=np.float32).copy()
        self.actions = np.asarray(actions, dtype=np.float32).copy()
        if (self.positions.ndim != 2 or self.positions.shape != (len(self.actions) + 1, 3)
                or self.actions.ndim != 2 or self.actions.shape[1] != 4
                or not np.isfinite(self.positions).all() or not np.isfinite(self.actions).all()):
            raise ValueError("Invalid AutoSERL reference trajectory")
        if (config.th1 <= 0 or config.th2 <= config.th1 or config.l_term <= 0
                or config.l_stag < 2 or config.step_size <= 0):
            raise ValueError("Require 0 < th1 < th2, l_term > 0, l_stag >= 2 and step_size > 0")
        contact = config.recover_point1
        if contact is None:
            closes = np.flatnonzero(self.actions[:, 3] > 1.5)
            contact = int(closes[0]) if len(closes) else len(self.actions) - 1
        if not 0 <= config.recover_point0 < contact < len(self.actions):
            raise ValueError("Recovery indices must satisfy 0 <= recover_point0 < recover_point1 < demo length")
        self.config = config
        self.safe_index = config.recover_point0
        self.contact_index = contact
        self.window_length = contact - self.safe_index + 1
        self.enabled = True
        self.observation = None
        self.reference = self.positions.copy()
        self.history: deque[np.ndarray] = deque(maxlen=config.l_stag)
        self.window_start = 0
        self.target_index: int | None = None
        self.recovery_index: int | None = None
        self.intervention_steps = 0
        self.recovery_armed = True

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.observation = observation
        self.history.clear()
        self.window_start = 0
        self.target_index = None
        self.recovery_index = None
        self.intervention_steps = 0
        self.recovery_armed = True
        self.reference = self.positions.copy()
        try:
            block = np.asarray(self.unwrapped.data.sensor("block_pos").data)
        except (AttributeError, KeyError):
            pass
        else:
            self.reference[:, :2] += block[:2] - self.positions[self.contact_index, :2]
        return observation, info

    def _guide(self, index: int, tcp: np.ndarray) -> np.ndarray:
        target = self.reference[index]
        try:
            controller_position = np.asarray(self.unwrapped.data.mocap_pos[0])
        except AttributeError:
            controller_position = tcp
        return np.array([*np.clip((target - controller_position) / self.config.step_size, -1, 1),
                         self.actions[index, 3]], dtype=np.float32)

    def step(self, action):
        if self.observation is None:
            raise RuntimeError("Reset the AutoSERL environment before stepping")
        tcp = np.asarray(self.observation["agent_pos"][-3:], dtype=np.float32)
        proposed = np.asarray(action, dtype=np.float32)
        executed = proposed
        mode = None
        if self.enabled:
            self.history.append(tcp.copy())
            nearest = int(np.argmin(np.linalg.norm(self.reference - tcp, axis=1)))
            if (self.recovery_index is None and self.recovery_armed and len(self.history) == self.config.l_stag
                    and self.safe_index <= nearest <= self.contact_index
                    and np.linalg.norm(self.history[-1] - self.history[0]) < self.config.th1):
                self.recovery_index = self.safe_index
                self.target_index = None
                self.recovery_armed = False
                self.history.clear()
            if self.recovery_index is not None:
                if np.linalg.norm(tcp - self.reference[self.recovery_index]) < self.config.th1:
                    if self.recovery_index == self.contact_index:
                        self.recovery_index = None
                        self.history.clear()
                    else:
                        self.recovery_index += 1
                if self.recovery_index is not None:
                    executed = self._guide(self.recovery_index, tcp)
                    mode = "recovery"
            else:
                if self.target_index is not None and np.linalg.norm(
                        tcp - self.reference[self.target_index]) < self.config.th1:
                    self.window_start = max(self.window_start, self.target_index)
                    self.target_index = None
                if self.target_index is None:
                    end = min(self.window_start + self.window_length, len(self.actions))
                    indices = np.arange(self.window_start, end)
                    distances = np.linalg.norm(self.reference[indices] - tcp, axis=1)
                    candidate = int(indices[np.argmin(distances)])
                    direction = self.reference[min(nearest + 1, len(self.reference) - 1)] - self.reference[nearest]
                    displacement = self.reference[candidate] - tcp
                    if distances.min() > self.config.th2 and np.dot(direction, displacement) >= 0:
                        self.target_index = candidate
                if self.target_index is not None:
                    executed = self._guide(self.target_index, tcp)
                    mode = "guidance"
                else:
                    self.window_start = min(self.window_start + 1,
                                            max(0, len(self.actions) - self.window_length))
        observation, reward, terminated, truncated, info = self.env.step(executed)
        self.observation = observation
        info = dict(info)
        human = bool(info.get("is_intervention", False))
        if mode is not None and not human:
            info["teleop_action"] = executed.copy()
            info["is_intervention"] = True
            info["auto_intervention"] = mode
            self.intervention_steps += 1
        if terminated or truncated:
            if (self.enabled and reward > 0 and self.intervention_steps < self.config.l_term):
                self.enabled = False
            info["auto_intervention_disabled"] = not self.enabled
        return observation, reward, terminated, truncated, info
