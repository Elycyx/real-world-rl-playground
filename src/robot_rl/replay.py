"""Online/offline replay with explicit transition semantics and compact image storage."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Transition:
    observation: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_observation: dict[str, np.ndarray]
    done: bool
    penalty: float = 0.0
    intervention: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.items: list[Transition] = []
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, transition: Transition) -> None:
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.cursor] = transition
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, count: int) -> list[Transition]:
        if not self.items:
            raise ValueError("Cannot sample an empty replay buffer")
        return [self.items[index] for index in self.rng.integers(len(self.items), size=count)]

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "cursor": self.cursor, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity must match the checkpoint when resuming")
        self.cursor = state["cursor"]
        self.rng.bit_generator.state = state["rng"]


def batch(online: ReplayBuffer, offline: ReplayBuffer, batch_size: int, device: torch.device) -> dict:
    if batch_size % 2:
        raise ValueError("50/50 replay requires an even batch size")
    items = online.sample(batch_size // 2) + offline.sample(batch_size // 2)
    observations = {key: torch.as_tensor(np.stack([item.observation[key] for item in items]), device=device)
                    for key in ("state", "front", "wrist")}
    next_observations = {key: torch.as_tensor(np.stack([item.next_observation[key] for item in items]), device=device)
                         for key in ("state", "front", "wrist")}
    return {
        "obs": observations, "next_obs": next_observations,
        "action": torch.as_tensor(np.stack([item.action for item in items]), device=device),
        "reward": torch.tensor([item.reward for item in items], device=device, dtype=torch.float32),
        "done": torch.tensor([item.done for item in items], device=device, dtype=torch.float32),
        "penalty": torch.tensor([item.penalty for item in items], device=device, dtype=torch.float32),
    }


def save_demos(path: Path, items: list[Transition]) -> None:
    if not items:
        raise ValueError("No transitions to save")
    arrays = {}
    for prefix, field in (("obs", "observation"), ("next_obs", "next_observation")):
        for key in ("state", "front", "wrist"):
            arrays[f"{prefix}_{key}"] = np.stack([getattr(item, field)[key] for item in items])
    for key in ("action", "reward", "done", "penalty", "intervention"):
        arrays[key] = np.asarray([getattr(item, key) for item in items])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_demos(path: Path, replay: ReplayBuffer) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
        for index in range(len(arrays["reward"])):
            replay.add(Transition(
                {key: arrays[f"obs_{key}"][index].copy() for key in ("state", "front", "wrist")},
                arrays["action"][index].astype(np.float32), float(arrays["reward"][index]),
                {key: arrays[f"next_obs_{key}"][index].copy() for key in ("state", "front", "wrist")},
                bool(arrays["done"][index]), float(arrays["penalty"][index]),
                bool(arrays["intervention"][index]),
            ))
