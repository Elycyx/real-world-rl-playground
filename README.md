# Real-world manipulation RL playground (MuJoCo)

This repository starts with **Panda PickCube** from `gym_hil` (MuJoCo) and a PyTorch implementation of the HIL-SERL/RLPD training loop. No real robot is required. The environment is the same Gym HIL task used in the [LeRobot HIL-SERL simulation tutorial](https://huggingface.co/docs/lerobot/main/en/hilserl_sim); the continuous SAC / discrete gripper split follows [LeRobot's PyTorch SAC](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rl/algorithms/sac/sac_algorithm.py), and the online/offline sampling and update order follow [the original HIL-SERL training script](https://github.com/rail-berkeley/hil-serl/blob/main/examples/train_rlpd.py).

## Setup

Install [uv](https://docs.astral.sh/uv/) first. The project uses Python 3.10 from `.python-version`. A graphical desktop with MuJoCo viewer support and a keyboard or gamepad are required for both demonstration collection and training. From the repository root, create/synchronize the environment (including the official pretrained vision model and test dependencies):

```
uv sync --locked --extra pretrained --extra test
```

The default pretrained vision model loads the **pinned** `lerobot/resnet10` revision from Hugging Face; this downloads executable model code (`trust_remote_code=True`) and weights. Only use the trusted official repository. `--scratch-vision` uses randomly initialized features and is not reference-equivalent.

If model loading reports `Unknown scheme for proxy URL ... socks://...`, this is a proxy setting, not a training error. HTTPX does not accept the legacy `socks://` scheme. If `HTTP_PROXY` and `HTTPS_PROXY` already point to working HTTP proxies, unset the redundant SOCKS fallback **for this command only**:

```
env -u ALL_PROXY -u all_proxy uv run --locked --extra pretrained robot-rl train --demos data/planned_pick_cube.npz --steps 100000 --actor-priority --queue-size 256     --batch-size 32
```

If you only have a SOCKS proxy, configure its URL with `socks5://` and install `httpx[socks]` instead. The pretrained model must still be downloaded on first use; `--scratch-vision` skips that download but changes the experiment's visual encoder.

```
uv run --locked --extra pretrained robot-rl collect --output data/pick_cube_demos.npz --steps 5000
uv run --locked --extra pretrained robot-rl collect --source motion-planning --episodes 20 --steps 10000 --output data/planned_pick_cube.npz
uv run --locked --extra pretrained robot-rl train --demos data/pick_cube_demos.npz --steps 100000
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --extra pretrained --extra test pytest -q
```

The viewer and intervention controller are always enabled in both commands. The default keyboard layout is:

- **Space**: toggle human takeover on/off (printed in the terminal; held-key repeats do not toggle repeatedly).
- **W/S**: forward/backward (X); **A/D**: left/right (Y); **Q/E**: raise/lower (Z). Hold keys for continuous movement.
- **G**: toggle the physical gripper open/closed with one tap (holding G does not repeatedly toggle; taps between control steps are still registered). **Enter**: mark success and end the episode; **Backspace/Esc**: mark failure and end the episode.

Keyboard input uses a system-wide listener: avoid pressing episode-end keys while typing elsewhere. Releasing motion keys stops human movement, but takeover stays active until Space is pressed again; gamepad control remains unchanged under `--gamepad`. The default `collect --source teleop` keeps **only human override** transitions; keep takeover active while guiding the robot. `collect --source motion-planning` instead plans Cartesian approach/grasp/lift waypoints from the MuJoCo cube pose, follows them with feedback from the robot state, and saves **complete successful episodes** (discarding failed or unfinished ones). `--episodes` stops after that many successes; `--steps` limits total attempts. This task-specific waypoint planner maintains clearance above the table but is **not** a collision-aware joint-space/OMPL planner. It uses privileged cube coordinates **only during collection**; the saved observation and trained policy still use the two cameras and robot state. Manual override remains possible during planned rollouts, and recorded actions always reflect what was executed. Planner steps are not marked as human interventions.

In keyboard mode, the always-visible MuJoCo scene is displayed in a dedicated GLFW window that intentionally has **no viewer keyboard shortcuts**. Mouse drag still rotates/translates the camera and the wheel zooms; keyboard input goes only to the intervention controller, even when this window has focus. Gamepad mode continues to use gym-hil's standard passive viewer. Closing the keyboard viewer stops the interactive run rather than silently continuing without a viewer.

Episodes now default to **500 actions** (`--max-episode-steps`), not a fixed number of wall-clock seconds. `gym_hil` registers PickCube with a 100-action limit; we override that limit at `gym.make` so the configured value actually takes effect. Interactive rollout and collection default to `--control-hz 10`, matching the environment's 0.1-second control timestep, so 500 steps take **about 50 seconds or longer** of wall time (slow inference/learning makes them longer). For more takeover time, use `--max-episode-steps 1000` for about 100 seconds or longer at 10 Hz. `--control-hz 0` disables pacing if you intentionally want faster-than-real-time simulation. Early success, failures, or manual episode termination still end an episode before the limit. `--steps` remains the **total** training/collection step budget, not a per-episode limit.

Training collects autonomous rollouts and adds live human overrides to the offline buffer alongside the precollected demonstrations (teleoperated or planned). The data file is a local NumPy archive, not a LeRobot dataset: export/convert external demonstrations to this schema before use. `--device cpu` and `--batch-size 4 --capacity 1000` enable a slower low-memory trial. Evaluation without interventions requires a separate evaluation-only task adapter; training always provides a viewer and controller.

## Distributed Actor–Learner

`robot-rl train` starts two local processes using Python's **spawn** start method: the command process is the MuJoCo Actor, and a child process is the PyTorch Learner. The Actor holds only a policy replica; it sends *executed* actions and transitions (including human overrides) through a bounded queue. The Learner exclusively owns the online/offline replay pools, optimizers, and checkpoints. It continuously updates once enough online transitions have arrived, including when the actor queue is briefly empty, and pushes the latest policy weights to the Actor every `--sync-every` learner updates (default: 50). The Actor takes the newest available weights before each action. `--queue-size` (default: 32) applies backpressure rather than silently dropping transitions. The default Actor device is CPU; the Learner uses CUDA when available (`--actor-device` and `--device` override them).

Startup prints JSON events with `role: startup`, `process` and `stage`: model loading, demonstration loading, actor weight transfer, and viewer opening. `Loading weights: 100%` only means the pretrained vision model is loaded; the Learner still needs to load demonstrations before the Actor starts. Demonstration NPZ columns are decompressed once during loading. If startup stops progressing, the last `stage` indicates which operation needs investigation.

For interactive responsiveness, the Actor now synchronizes only the *trainable* encoder/head weights when using the pretrained, frozen ResNet10; both processes load the same pinned frozen backbone at startup. Every `--actor-log-every` steps (default: 50), `actor_timing.jsonl` and stdout report mean and worst-case milliseconds for `sync` (parameter transfer), `policy` (action inference), `environment` (MuJoCo step, camera render and controller polling), and `queue` (transition handoff). A `queue_backpressure` warning means the Learner has fallen behind and the bounded queue is pausing control. `--actor-priority` drains the transition queue before each Learner update, keeping the Actor moving when updates are slow; it delays weight updates while the Actor is busy but still completes the configured update budget after rollout (unless interrupted). If `queue` dominates, try `--actor-priority` with a larger `--queue-size` (temporary buffering) or smaller `--batch-size`; if `environment` dominates, check viewer/rendering performance; if `policy` dominates, check Actor CPU load. Smaller batches change the learning dynamics and a bigger queue cannot fix persistent Learner under-capacity. For smoother operation when training and rendering share a GPU, try `--actor-priority --batch-size 32 --queue-size 256 --actor-log-every 25`; compare timing logs before and after rather than assuming any one setting fixes the bottleneck.

`--steps` is the Actor's *total* environment-step target; `--learner-updates` is the Learner's *total* update target (default: the same as `--steps`). The Learner drains queued transitions after Actor shutdown, completes its remaining update budget, and saves the final checkpoint. Ctrl-C asks the Learner to drain and save before the processes exit. Logs include `role` and process `pid` to distinguish the two loops. This is local inter-process distribution, **not** multi-machine RPC; if the Learner cannot keep up, the bounded queue will pause the Actor.

### Online training metrics

Each `train` invocation appends JSON Lines in its `--output` directory (no extra dependencies):

- `sessions.jsonl`: UTC start/end timestamps, wall-clock seconds (including initialization, rollout, learner drain and checkpoint), task, algorithm, seed, demonstration paths, resume path, and full CLI configuration.
- `actor_metrics.jsonl`: one record per **completed** episode with total environment step, episode length/reward/duration, cumulative actor elapsed seconds and steps/second, success, timeout, total/human/automatic interventions, random warmup actions, cumulative success rate, and success rate over the last 20 completed episodes. `policy_success_rate` counts only episodes with **zero interventions and zero random warmup actions**; it is `null` until a qualifying episode completes. Overall success rate includes assisted episodes.
- `learner_metrics.jsonl`: first, last, and every `--log-every` updates, records total environment steps, total learner updates, update duration, learner elapsed seconds, per-session updates/second, replay sizes and algorithm losses; the final record contains an `event: finished` marker.

### Weights & Biases

All `train` algorithms (`hil_serl`, `auto_serl`, and future registry entries) also
write a W&B run by default in **offline mode**. No account or network connection is
required for offline recording. Use `--wandb-mode online --wandb-project PROJECT`
after `wandb login` to stream curves to W&B, or `--wandb-mode disabled` to turn it
off. `WANDB_PROJECT` can also select the project (default: `robot-rl-playground`).
The offline files are under `<output>/wandb/` and can be uploaded later with
`wandb sync <output>/wandb/offline-run-*`. Every training invocation, including
resume, creates its own run identified by `session_id` and grouped by task and
algorithm; configuration and seed are saved with it.

W&B charts include per-episode reward, success rates (overall, last 20, policy-only),
total/human/automatic interventions and recovery steps, random actions and episode
length; per-learner-update actor/critic/gripper losses, entropy temperature, replay
sizes and update rate; and Actor inference/environment/queue/sync latencies.
Episode and latency metrics use environment steps on the x-axis; learner metrics
use update count. The first and last learner update are logged even when the
run does not reach `--log-every` (default: 100). Existing JSONL logs, replay
archives and checkpoints remain available independently of W&B.

All records include a `session_id`; resume appends a **new** session rather than overwriting metrics, and rate counters restart per session. `step` and `updates` remain absolute after resume. Only completed episodes contribute to success rates; a partially collected episode at shutdown is excluded. These are online training measurements with the viewer and intervention enabled, **not** intervention-free evaluation results; compare `policy_success_rate` as an unassisted training indicator and use a separate controlled evaluation for final algorithm comparisons.

The reference example uses 128×128 front/wrist RGB, 18 robot-state values, three bounded Cartesian deltas and a three-choice gripper, sparse success rewards, 0.97 discount, 0.005 target smoothing, 0.01 initial entropy temperature, two critics, 50/50 online/offline batches, two critic updates and one actor/temperature update per learner step. Discrete gripper rewards include the environment's grip penalty. Time-limit truncations bootstrap while terminal states do not. The online buffer includes autonomous and overridden actions; the offline buffer includes demonstrations and live human overrides. The Learner saves its optimizers, policy and replay pools under `runs/pick_cube/hil_serl/`; resume with `--resume runs/pick_cube/hil_serl/checkpoint.pt` and the original `--demos`. Resume starts a fresh MuJoCo episode rather than restoring the exact physics state mid-episode.

## AutoSERL trajectory-guided training

`--algorithm auto_serl` reuses the existing PyTorch hybrid SAC/RLPD learner, replay
buffers, checkpointing and actor/learner processes. It adds an Actor-side automatic
intervention wrapper adapted from AutoSERL's sliding window intervention, stagnation
safety recovery and intervention termination. First collect **one successful, complete,
contiguous episode** as an `.npz` archive (for example with
`collect --source motion-planning --episodes 1`); this trajectory is the intervention reference. An
archive containing several episodes or only isolated human override steps is not a
valid trajectory. The same archive supplies the offline replay:

```bash
robot-rl collect --source motion-planning --episodes 1 --output data/auto_demo.npz --steps 5000
robot-rl train --algorithm auto_serl --demos data/auto_demo.npz \
  --output runs/pick_cube/auto_serl --steps 100000
```

`--auto-demo` selects a separate single-episode reference if `--demos` contains
additional archives. `--th1` (0.005 m) completes guidance at a waypoint; `--th2`
(0.02 m) starts guidance when the nearest sliding-window waypoint is too far away.
`--l-stag` (20 steps) detects stagnation; `--recover-point0` (0) and
`--recover-point1` (first close-gripper step by default) bound a safe-to-contact
recovery segment. `--l-term` (10 guided steps) disables future automatic
intervention after a successful episode requiring fewer guided steps. Human
keyboard/gamepad actions take precedence; executed automatic actions enter both
online and offline replay through the existing intervention flag. The reference is
translated in XY to the current block position using its contact waypoint, so this
mode currently supports only `pick_cube`. PickCube controls translation and a
discrete gripper rather than the upstream Franka 6-DoF pose; recovery follows TCP
waypoints and the recorded gripper command rather than replaying 6-DoF actions.
Resume restarts the Actor's intervention state, including early termination; it
does not restore the physical episode.

## Extension points

- `src/robot_rl/registry.py`: register another task adapter or algorithm implementation for `--task`/`--algorithm` selection.
- `src/robot_rl/tasks/`: add a Gymnasium/MuJoCo adapter returning `state`, `front`, `wrist` (or implement a new observation encoder); keep `actual_action` for intervention overrides.
- `src/robot_rl/planners/`: register task-specific collection planners in `registry.py`; they generate actions through the same viewer/override-enabled task adapter as teleoperation.
- `src/robot_rl/algorithms/`: add an algorithm exposing `act`, `update`, `state_dict`, `load_state_dict`; replay does not depend on the algorithm.
- `src/robot_rl/replay.py`: common transitions, independent online/offline pools and portable demonstration archives.
- `src/robot_rl/distributed.py`: Actor/Learner process orchestration and queue protocol; `src/robot_rl/cli.py` defines experiment settings. For large runs, reduce capacity or replace the in-memory image replay with disk-backed storage (100k RGB transitions require many gigabytes).

**Current differences from the distributed original:** processes communicate through local bounded multiprocessing queues instead of networked servers; the custom PyTorch model uses LeRobot's pretrained ResNet10 with mean pooling and our own fusion MLP (not LeRobot's identical actor preprocessing). ImageNet image statistics and the fixed Min/Max state statistics from the official example are used; there is no reward classifier. Sparse MuJoCo task reward is available, so a learned reward classifier is unnecessary for this task. Do not interpret this as an exact numerical reproduction of the original JAX or LeRobot checkpoints. The pinned pretrained model has been validated locally for action selection and parameter updates.
