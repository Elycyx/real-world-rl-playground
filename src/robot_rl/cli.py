"""Collect human interventions and train HIL-SERL on MuJoCo PickCube."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from robot_rl.control import wait_for_tick
from robot_rl.distributed import train
from robot_rl.registry import ALGORITHMS, PLANNERS, TASKS
from robot_rl.replay import save_demos
from robot_rl.rollout import transition


def collect(args: argparse.Namespace) -> None:
    if args.source == "motion-planning" and args.task not in PLANNERS:
        raise ValueError(f"No motion planner is registered for task {args.task}")
    if args.episodes is not None and (args.source != "motion-planning" or args.episodes <= 0):
        raise ValueError("--episodes requires motion-planning and a positive count")
    if args.control_hz < 0:
        raise ValueError("--control-hz must be nonnegative")
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive")
    np.random.seed(args.seed)
    task = TASKS[args.task]
    env = task.make_env(gamepad=args.gamepad, seed=args.seed,
                   max_episode_steps=args.max_episode_steps)
    env.action_space.seed(args.seed)
    planner = PLANNERS[args.task]() if args.source == "motion-planning" else None
    rng = np.random.default_rng(args.seed)
    examples = []
    planned_episode = []
    successful_episodes = 0
    try:
        observation, _ = env.reset(seed=args.seed)
        if planner is not None:
            planner.reset(env, observation)
        next_tick = 0.0
        for step in range(args.steps):
            next_tick = wait_for_tick(next_tick, args.control_hz)
            if planner is None:
                action = env.action_space.sample().astype(np.float32)
                action[:3] = rng.uniform(-1, 1, size=3)
                action[3] = rng.integers(3)
            else:
                action = planner.act(env, observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            item = transition(observation, action, reward, next_observation, terminated, info, task)
            if planner is not None:
                planned_episode.append(item)
            elif item.intervention:
                examples.append(item)
            if terminated or truncated:
                if planner is not None:
                    success = bool(info.get("succeed", False) or info.get("next.success", False))
                    if success:
                        examples.extend(planned_episode)
                        successful_episodes += 1
                    planned_episode.clear()
                    print(f"episode at {step + 1}: success={success} "
                          f"successful_episodes={successful_episodes} collected={len(examples)}", flush=True)
                    if args.episodes is not None and successful_episodes >= args.episodes:
                        break
                else:
                    print(f"episode at {step + 1}: reward={reward} collected={len(examples)}", flush=True)
                observation, _ = env.reset()
                if planner is not None:
                    planner.reset(env, observation)
            else:
                observation = next_observation
    finally:
        env.close()
        if examples:
            save_demos(args.output, examples)
            print(f"saved {len(examples)} {args.source} transitions to {args.output}")
    if planner is not None and not examples:
        raise RuntimeError("No successful planned episodes; increase --steps or adjust planner/task settings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect", help="Record keyboard/gamepad overrides")
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--steps", type=int, default=5000)
    collect_parser.add_argument("--source", choices=("teleop", "motion-planning"), default="teleop")
    collect_parser.add_argument("--episodes", type=int, help="Successful motion-planned episodes to collect")
    train_parser = commands.add_parser("train", help="Train hybrid SAC with 50/50 replay")
    train_parser.add_argument("--demos", type=Path, nargs="+", required=True)
    train_parser.add_argument("--algorithm", choices=ALGORITHMS, default="hil_serl")
    train_parser.add_argument("--output", type=Path, default=Path("runs/pick_cube/hil_serl"))
    train_parser.add_argument("--steps", type=int, default=100_000)
    train_parser.add_argument("--learner-updates", type=int,
                              help="Independent learner update budget (defaults to --steps)")
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--capacity", type=int, default=100_000)
    train_parser.add_argument("--learning-starts", type=int, default=100)
    train_parser.add_argument("--random-steps", type=int, default=100)
    train_parser.add_argument("--utd-ratio", type=int, default=2)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--actor-log-every", type=int, default=50,
                              help="Actor steps between per-stage latency reports")
    train_parser.add_argument("--save-every", type=int, default=10_000)
    train_parser.add_argument("--sync-every", type=int, default=50,
                              help="Learner updates between actor parameter pushes")
    train_parser.add_argument("--queue-size", type=int, default=32,
                              help="Maximum in-flight transitions; backpressure prevents drops")
    train_parser.add_argument("--actor-priority", action="store_true",
                              help="Drain queued transitions before each learner update to prioritize control")
    train_parser.add_argument("--learner-threads", type=int, default=4)
    train_parser.add_argument("--actor-threads", type=int, default=1)
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--actor-device", default="cpu")
    train_parser.add_argument("--scratch-vision", action="store_true",
                              help="Use a randomly initialized trainable ResNet10 (not the reference setting)")
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline",
                              help="W&B run mode; offline records locally without login (default)")
    train_parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "robot-rl-playground"),
                              help="W&B project for all training algorithms")
    train_parser.add_argument("--auto-demo", type=Path,
                              help="Single successful demo trajectory for AutoSERL (defaults to first --demos)")
    train_parser.add_argument("--th1", type=float, default=0.005,
                              help="AutoSERL guidance completion threshold, meters")
    train_parser.add_argument("--th2", type=float, default=0.02,
                              help="AutoSERL guidance trigger threshold, meters")
    train_parser.add_argument("--l-term", type=int, default=10,
                              help="Disable automatic intervention after success with fewer guided steps")
    train_parser.add_argument("--l-stag", type=int, default=20,
                              help="AutoSERL stagnation detection window, steps")
    train_parser.add_argument("--recover-point0", type=int, default=0,
                              help="Index of safe recovery waypoint")
    train_parser.add_argument("--recover-point1", type=int,
                              help="Index of stable-contact waypoint (default: first close-gripper step)")
    for subparser in (collect_parser, train_parser):
        subparser.add_argument("--task", choices=TASKS, default="pick_cube")
        subparser.add_argument("--gamepad", action="store_true")
        subparser.add_argument("--seed", type=int, default=1000)
        subparser.add_argument("--max-episode-steps", type=int, default=500,
                               help="Per-episode action limit (not a wall-clock duration)")
        subparser.add_argument("--control-hz", type=float, default=10.0,
                               help="Maximum interactive action rate; 0 disables pacing")
    args = parser.parse_args()
    (collect if args.command == "collect" else train)(args)


if __name__ == "__main__":
    main()
