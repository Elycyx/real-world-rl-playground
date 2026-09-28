"""Spawned learner and independent MuJoCo actor with bounded transition transport."""

import json
import multiprocessing as mp
import os
import random
import signal
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full

import numpy as np
import torch

from robot_rl.algorithms.auto_serl import (
    AutoIntervention,
    AutoSerlConfig,
    load_reference,
)
from robot_rl.algorithms.hil_serl import HILSerlConfig
from robot_rl.control import wait_for_tick
from robot_rl.metrics import EpisodeMetrics, StepTimings, record
from robot_rl.registry import ACTORS, ALGORITHMS, TASKS
from robot_rl.replay import ReplayBuffer, batch, load_demos, save_demos
from robot_rl.rollout import transition
from robot_rl.tracking import WandbTracker


def save_checkpoint(path: Path, step: int, agent, online: ReplayBuffer,
                    offline: ReplayBuffer, updates: int = 0) -> None:
    checkpoint = {"algorithm": agent.state_dict(), "step": step, "updates": updates,
                  "online_replay": online.state_dict(), "offline_replay": offline.state_dict(),
                  "torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path / "checkpoint.pt")
    save_demos(path / "online.npz", online.items)
    save_demos(path / "offline.npz", offline.items)


def _publish(weights_queue, agent) -> None:
    while True:
        try:
            weights_queue.get_nowait()
        except Empty:
            break
    try:
        weights_queue.put_nowait(agent.actor_weights())
    except Full:
        pass


def _learner(args, transitions_queue, weights_queue, status_queue, stop_event, metrics_queue) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        torch.set_num_threads(args.learner_threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        print(json.dumps({"role": "startup", "process": "learner", "pid": os.getpid(),
                          "stage": "loading_model"}), flush=True)
        algorithm = ALGORITHMS[args.algorithm](HILSerlConfig(utd_ratio=args.utd_ratio), device,
                                               not args.scratch_vision)
        print(json.dumps({"role": "startup", "process": "learner", "pid": os.getpid(),
                          "stage": "model_loaded"}), flush=True)
        online = ReplayBuffer(args.capacity, args.seed)
        offline = ReplayBuffer(args.capacity, args.seed + 1)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            algorithm.load_state_dict(checkpoint["algorithm"])
            step = checkpoint["step"]
            updates = checkpoint.get("updates", max(0, step - args.learning_starts + 1))
            load_demos(args.resume.with_name("online.npz"), online)
            load_demos(args.resume.with_name("offline.npz"), offline)
            if "online_replay" in checkpoint:
                online.load_state_dict(checkpoint["online_replay"])
                offline.load_state_dict(checkpoint["offline_replay"])
            if "torch_rng" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng"].cpu())
            if "cuda_rng" in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        else:
            step = 0
            updates = 0
            for demo in args.demos:
                print(json.dumps({"role": "startup", "process": "learner", "pid": os.getpid(),
                                  "stage": "loading_demos",
                                  "path": str(demo)}), flush=True)
                load_demos(demo, offline)
            print(json.dumps({"role": "startup", "process": "learner", "pid": os.getpid(),
                              "stage": "demos_loaded",
                              "transitions": len(offline)}), flush=True)
        if not len(offline):
            raise ValueError("Supply human demonstrations via --demos; offline replay cannot be empty")
        args.output.mkdir(parents=True, exist_ok=True)
        _publish(weights_queue, algorithm)
        status_queue.put(("ready", step))

        target_updates = args.learner_updates if args.learner_updates is not None else args.steps
        actor_finished = False
        start_updates = updates
        learner_start = time.perf_counter()
        while True:
            if actor_finished and (updates >= target_updates or len(online) < args.learning_starts):
                break
            if not actor_finished:
                try:
                    item = transitions_queue.get(timeout=0.05)
                except Empty:
                    if stop_event.is_set():
                        break
                    item = False
                if item is None:
                    actor_finished = True
                elif item is not False:
                    for _ in range(args.queue_size + 1):
                        if item is None:
                            actor_finished = True
                            break
                        online.add(item)
                        if item.intervention:
                            offline.add(item)
                        step += 1
                        if step % args.save_every == 0 and not stop_event.is_set():
                            save_checkpoint(args.output, step, algorithm, online, offline, updates)
                        if not args.actor_priority:
                            break
                        try:
                            item = transitions_queue.get_nowait()
                        except Empty:
                            break
            if len(online) >= args.learning_starts and updates < target_updates and not stop_event.is_set():
                update_start = time.perf_counter()
                stats = algorithm.update(lambda: batch(online, offline, args.batch_size, device))
                update_duration = time.perf_counter() - update_start
                updates += 1
                if updates % args.sync_every == 0:
                    _publish(weights_queue, algorithm)
                if (updates == start_updates + 1 or updates % args.log_every == 0
                        or updates == target_updates):
                    event = {"role": "learner", "pid": os.getpid(), "session_id": args.session_id,
                             "step": step, "updates": updates, "online": len(online),
                             "offline": len(offline), "learner_elapsed_s": time.perf_counter() - learner_start,
                             "update_duration_s": update_duration, **stats}
                    event["updates_per_second"] = (updates - start_updates) / max(event["learner_elapsed_s"], 1e-9)
                    record(args.output / "learner_metrics.jsonl", **event)
                    if metrics_queue is not None:
                        metrics_queue.put(event)
                    print(json.dumps(event), flush=True)

        if len(online):
            save_checkpoint(args.output, step, algorithm, online, offline, updates)
        record(args.output / "learner_metrics.jsonl", role="learner", session_id=args.session_id,
               event="finished", step=step, updates=updates,
               learner_elapsed_s=time.perf_counter() - learner_start)
        status_queue.put(("finished", step))
    except (RuntimeError, ValueError, OSError, LookupError, TypeError):
        status_queue.put(("error", traceback.format_exc()))
    finally:
        weights_queue.cancel_join_thread()


def _status(status_queue, learner, expected: str, weights_queue=None):
    while True:
        if weights_queue is not None:
            while True:
                try:
                    weights_queue.get_nowait()
                except Empty:
                    break
        try:
            kind, value = status_queue.get(timeout=0.25)
        except Empty:
            if not learner.is_alive():
                raise RuntimeError(f"Learner exited unexpectedly (exit code {learner.exitcode})") from None
            continue
        if kind == "error":
            raise RuntimeError(f"Learner failed:\n{value}")
        if kind != expected:
            raise RuntimeError(f"Unexpected learner status: {kind}")
        return value


def _check_learner(status_queue, learner) -> None:
    try:
        kind, value = status_queue.get_nowait()
    except Empty:
        if not learner.is_alive():
            raise RuntimeError(f"Learner exited unexpectedly (exit code {learner.exitcode})") from None
        return
    if kind == "error":
        raise RuntimeError(f"Learner failed:\n{value}")
    raise RuntimeError(f"Learner stopped before actor finished: {kind} ({value})")


def _send(transitions_queue, item, status_queue, learner, report_backpressure: bool = False) -> None:
    last_warning = time.perf_counter()
    while True:
        _check_learner(status_queue, learner)
        try:
            transitions_queue.put(item, timeout=0.25)
            return
        except Full:
            if report_backpressure and time.perf_counter() - last_warning >= 2:
                print(json.dumps({"role": "actor", "event": "queue_backpressure",
                                  "message": "Actor is waiting for the Learner to drain transitions"}), flush=True)
                last_warning = time.perf_counter()
            continue


def _actor(args, start_step: int, transitions_queue, weights_queue, status_queue, learner,
           metrics_queue) -> None:
    torch.set_num_threads(args.actor_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    task = TASKS[args.task]
    device = torch.device(args.actor_device)
    print(json.dumps({"role": "startup", "process": "actor", "pid": os.getpid(),
                      "stage": "loading_model"}), flush=True)
    policy = ACTORS[args.algorithm](HILSerlConfig(utd_ratio=args.utd_ratio), device,
                                    not args.scratch_vision)
    print(json.dumps({"role": "startup", "process": "actor", "pid": os.getpid(),
                      "stage": "model_loaded"}), flush=True)
    print(json.dumps({"role": "startup", "process": "actor", "pid": os.getpid(),
                      "stage": "waiting_for_weights"}), flush=True)
    while True:
        _check_learner(status_queue, learner)
        try:
            policy.load_weights(weights_queue.get(timeout=0.25))
            break
        except Empty:
            continue
    print(json.dumps({"role": "startup", "process": "actor", "pid": os.getpid(),
                      "stage": "weights_received"}), flush=True)

    if start_step >= args.steps:
        return
    print(json.dumps({"role": "startup", "process": "actor", "pid": os.getpid(),
                      "stage": "opening_viewer"}), flush=True)
    env = task.make_env(gamepad=args.gamepad, seed=args.seed,
                        max_episode_steps=args.max_episode_steps)
    try:
        if args.algorithm == "auto_serl":
            positions, actions = load_reference(args.auto_demo or args.demos[0])
            env = AutoIntervention(env, positions, actions, AutoSerlConfig(
                th1=args.th1, th2=args.th2, l_term=args.l_term, l_stag=args.l_stag,
                recover_point0=args.recover_point0, recover_point1=args.recover_point1))
    except (OSError, ValueError, KeyError):
        env.close()
        raise
    env.action_space.seed(args.seed)
    try:
        observation, _ = env.reset(seed=args.seed)
        print(json.dumps({"role": "startup", "process": "actor", "pid": os.getpid(),
                          "stage": "viewer_ready"}), flush=True)
        metrics = EpisodeMetrics(start_step)
        timings = StepTimings()
        next_tick = 0.0
        for step in range(start_step, args.steps):
            next_tick = wait_for_tick(next_tick, args.control_hz)
            _check_learner(status_queue, learner)
            started = time.perf_counter()
            while True:
                try:
                    policy.load_weights(weights_queue.get_nowait())
                except Empty:
                    break
            synced = time.perf_counter()
            if step < args.random_steps:
                action = env.action_space.sample().astype(np.float32)
                action[3] = np.random.randint(3)
            else:
                action = policy.act(task.pack_observation(observation)).numpy()
            acted = time.perf_counter()
            next_observation, reward, terminated, truncated, info = env.step(action)
            item = transition(observation, action, reward, next_observation, terminated, info, task)
            transitioned = time.perf_counter()
            _send(transitions_queue, item, status_queue, learner, report_backpressure=True)
            sent = time.perf_counter()
            timings.add(synced - started, acted - synced, transitioned - acted, sent - transitioned)
            if timings.count >= args.actor_log_every:
                event = {"role": "actor", "event": "timing", "pid": os.getpid(),
                         "session_id": args.session_id, **timings.flush(step + 1)}
                record(args.output / "actor_timing.jsonl", **event)
                if metrics_queue is not None:
                    metrics_queue.put(event)
                print(json.dumps(event), flush=True)
            metrics.add(reward, item.intervention, step < args.random_steps, info.get("auto_intervention"))
            if terminated or truncated:
                event = {"role": "actor", "pid": os.getpid(), "session_id": args.session_id,
                         **metrics.finish(step + 1,
                                          bool(info.get("succeed", False) or info.get("next.success", False)),
                                          truncated)}
                event["steps_per_second"] = (step + 1 - start_step) / max(event["actor_elapsed_s"], 1e-9)
                record(args.output / "actor_metrics.jsonl", **event)
                if metrics_queue is not None:
                    metrics_queue.put(event)
                print(json.dumps(event), flush=True)
                observation, _ = env.reset()
                metrics.reset(step + 1)
            else:
                observation = next_observation
        if timings.count:
            event = {"role": "actor", "event": "timing", "pid": os.getpid(),
                     "session_id": args.session_id, **timings.flush(args.steps)}
            record(args.output / "actor_timing.jsonl", **event)
            if metrics_queue is not None:
                metrics_queue.put(event)
    finally:
        env.close()


def train(args) -> None:
    if args.algorithm == "auto_serl" and args.task != "pick_cube":
        raise ValueError("AutoSERL trajectory guidance currently supports only pick_cube")
    if args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError("--batch-size must be positive and even")
    for name in ("capacity", "learning_starts", "utd_ratio", "log_every", "actor_log_every",
                 "save_every", "sync_every",
                 "queue_size", "learner_threads", "actor_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.steps < 0 or args.random_steps < 0:
        raise ValueError("--steps and --random-steps must be nonnegative")
    if args.learner_updates is not None and args.learner_updates < 0:
        raise ValueError("--learner-updates must be nonnegative")
    if args.max_episode_steps <= 0 or args.control_hz < 0:
        raise ValueError("--max-episode-steps must be positive and --control-hz must be nonnegative")
    args.session_id = uuid.uuid4().hex
    training_start = time.perf_counter()
    session_path = args.output / "sessions.jsonl"
    config = {key: [str(item) for item in value] if isinstance(value, list)
              else str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "session_id"}
    args.output.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    metrics_queue = context.Queue() if args.wandb_mode != "disabled" else None
    tracker = WandbTracker(args, config, metrics_queue) if metrics_queue is not None else None
    record(session_path, event="started", session_id=args.session_id,
           started_utc=datetime.now(timezone.utc).isoformat(), task=args.task,
           algorithm=args.algorithm, seed=args.seed, demos=[str(path) for path in args.demos],
           resume=str(args.resume) if args.resume else None, config=config)
    transitions_queue = context.Queue(maxsize=args.queue_size)
    weights_queue = context.Queue(maxsize=1)
    status_queue = context.Queue()
    stop_event = context.Event()
    learner = context.Process(target=_learner, args=(args, transitions_queue, weights_queue,
                                                     status_queue, stop_event, metrics_queue),
                              name=f"{args.algorithm.replace('_', '-')}-learner")
    succeeded = False
    try:
        learner.start()
        start_step = _status(status_queue, learner, "ready")
        _actor(args, start_step, transitions_queue, weights_queue, status_queue, learner, metrics_queue)
        _send(transitions_queue, None, status_queue, learner)
        completed_step = _status(status_queue, learner, "finished", weights_queue)
        learner.join(timeout=30)
        if learner.is_alive() or learner.exitcode != 0 or completed_step != max(start_step, args.steps):
            raise RuntimeError(f"Learner did not finish cleanly at step {args.steps}")
        succeeded = True
    except BaseException:
        stop_event.set()
        if learner.pid is not None:
            learner.join(timeout=30)
            if learner.is_alive():
                learner.terminate()
                learner.join()
        raise
    finally:
        try:
            if tracker is not None:
                tracker.finish(succeeded)
        finally:
            record(session_path, event="finished" if succeeded and (tracker is None or tracker.error is None)
                   else "failed", session_id=args.session_id,
                   ended_utc=datetime.now(timezone.utc).isoformat(),
                   wall_time_s=time.perf_counter() - training_start)
            for queue in (transitions_queue, weights_queue, status_queue):
                queue.cancel_join_thread()
                queue.close()
            if metrics_queue is not None:
                metrics_queue.close()
                metrics_queue.join_thread()
