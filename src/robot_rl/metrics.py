"""Append-only, task-agnostic measurements for online training runs."""

import json
import time
from collections import deque
from pathlib import Path


def record(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(values) + "\n")


class StepTimings:
    def __init__(self):
        self.count = 0
        self.durations = {stage: [] for stage in ("sync", "policy", "environment", "queue")}

    def add(self, sync: float, policy: float, environment: float, queue: float) -> None:
        self.count += 1
        for stage, duration in (("sync", sync), ("policy", policy),
                                ("environment", environment), ("queue", queue)):
            self.durations[stage].append(duration * 1000)

    def flush(self, step: int) -> dict:
        result = {"step": step, "window_steps": self.count}
        for stage, durations in self.durations.items():
            result[f"{stage}_mean_ms"] = sum(durations) / self.count
            result[f"{stage}_max_ms"] = max(durations)
            durations.clear()
        self.count = 0
        return result


class EpisodeMetrics:
    def __init__(self, start_step: int):
        self.episode_start_step = start_step
        self.start_time = time.perf_counter()
        self.episode_start_time = self.start_time
        self.episodes = 0
        self.successes = 0
        self.policy_episodes = 0
        self.policy_successes = 0
        self.recent_successes = deque(maxlen=20)
        self.reward = 0.0
        self.interventions = 0
        self.auto_interventions = 0
        self.recovery_steps = 0
        self.random_actions = 0

    def add(self, reward: float, intervention: bool, random_action: bool,
            auto_intervention: str | None = None) -> None:
        self.reward += reward
        self.interventions += int(intervention)
        self.auto_interventions += int(auto_intervention is not None)
        self.recovery_steps += int(auto_intervention == "recovery")
        self.random_actions += int(random_action)

    def finish(self, step: int, success: bool, truncated: bool) -> dict:
        now = time.perf_counter()
        self.episodes += 1
        self.successes += int(success)
        self.recent_successes.append(success)
        policy_only = self.interventions == 0 and self.random_actions == 0
        if policy_only:
            self.policy_episodes += 1
            self.policy_successes += int(success)
        return {
            "step": step,
            "episode": self.episodes,
            "episode_steps": step - self.episode_start_step,
            "episode_duration_s": now - self.episode_start_time,
            "actor_elapsed_s": now - self.start_time,
            "episode_reward": self.reward,
            "success": bool(success),
            "truncated": bool(truncated),
            "interventions": self.interventions,
            "auto_interventions": self.auto_interventions,
            "human_interventions": self.interventions - self.auto_interventions,
            "recovery_steps": self.recovery_steps,
            "random_actions": self.random_actions,
            "success_rate": self.successes / self.episodes,
            "success_rate_last_20": sum(self.recent_successes) / len(self.recent_successes),
            "policy_episodes": self.policy_episodes,
            "policy_success_rate": (
                self.policy_successes / self.policy_episodes
                if self.policy_episodes else None
            ),
        }

    def reset(self, step: int) -> None:
        self.episode_start_step = step
        self.episode_start_time = time.perf_counter()
        self.reward = 0.0
        self.interventions = 0
        self.auto_interventions = 0
        self.recovery_steps = 0
        self.random_actions = 0
