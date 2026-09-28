import numpy as np
import pytest
import torch

from robot_rl.algorithms.hil_serl import (
    ActorReplica,
    HILSerl,
    HILSerlConfig,
    VisualEncoder,
)
from robot_rl.cli import transition
from robot_rl.replay import ReplayBuffer, batch, load_demos, save_demos
from robot_rl.tasks import pick_cube


def observation():
    return {"agent_pos": np.zeros(18, dtype=np.float32),
            "pixels": {key: np.zeros((32, 32, 3), dtype=np.uint8) for key in ("front", "wrist")}}


def test_pretrained_encoder_reports_invalid_proxy(monkeypatch):
    from transformers import AutoModel

    def fail_download(*args, **kwargs):
        raise OSError("Unknown scheme for proxy URL URL('socks://127.0.0.1:7890/')")

    monkeypatch.setattr(AutoModel, "from_pretrained", fail_download)
    with pytest.raises(RuntimeError, match="env -u ALL_PROXY -u all_proxy"):
        VisualEncoder(latent_dim=16, hidden_dim=32, pretrained=True)


def test_pretrained_encoder_preserves_other_download_errors(monkeypatch):
    from transformers import AutoModel

    def fail_download(*args, **kwargs):
        raise OSError("Model weights unavailable")

    monkeypatch.setattr(AutoModel, "from_pretrained", fail_download)
    with pytest.raises(OSError, match="Model weights unavailable"):
        VisualEncoder(latent_dim=16, hidden_dim=32, pretrained=True)


def test_intervention_stores_executed_action_and_timeout_bootstraps(tmp_path):
    original_observation = observation()
    item = transition(original_observation, np.zeros(4, dtype=np.float32), 1.0, observation(),
                      False, {"teleop_action": np.array([1, 0, 0, 2], dtype=np.float32),
                              "is_intervention": True, "discrete_penalty": -0.02})
    original_observation["agent_pos"][:] = 2
    original_observation["pixels"]["front"][:] = 255
    assert item.observation["state"].sum() == 0
    assert item.observation["front"].sum() == 0
    online = ReplayBuffer(2)
    offline = ReplayBuffer(2)
    online.add(item)
    save_demos(tmp_path / "demos.npz", [item])
    load_demos(tmp_path / "demos.npz", offline)
    sampled = batch(online, offline, 2, torch.device("cpu"))
    assert sampled["action"].tolist() == [[1, 0, 0, 2], [1, 0, 0, 2]]
    assert sampled["done"].tolist() == [0, 0]
    np.testing.assert_allclose(sampled["penalty"].tolist(), [-0.02, -0.02])
    assert offline.items[0].intervention


def test_load_demos_decompresses_each_array_once(tmp_path, monkeypatch):
    from robot_rl import replay as replay_module

    original_load = replay_module.np.load
    reads = {}

    class CountingArchive:
        def __init__(self, path, **kwargs):
            self.archive = original_load(path, **kwargs)
            self.files = self.archive.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.archive.__exit__(*args)

        def __getitem__(self, key):
            reads[key] = reads.get(key, 0) + 1
            return self.archive[key]

    archive = tmp_path / "demos.npz"
    item = transition(observation(), np.array([0, 0, 0, 1], dtype=np.float32),
                      1.0, observation(), True, {})
    save_demos(archive, [item] * 10)
    monkeypatch.setattr(replay_module.np, "load", CountingArchive)
    replay = ReplayBuffer(10)
    load_demos(archive, replay)
    assert len(replay) == 10
    assert set(reads) == {"obs_state", "obs_front", "obs_wrist", "next_obs_state",
                          "next_obs_front", "next_obs_wrist", "action", "reward", "done",
                          "penalty", "intervention"}
    assert all(count == 1 for count in reads.values())
    assert replay.items[0].reward == 1.0


def test_pytorch_update_and_hybrid_action():
    torch.set_num_threads(1)
    device = torch.device("cpu")
    config = HILSerlConfig(hidden_dim=32, latent_dim=16, utd_ratio=2)
    agent = HILSerl(config, device, pretrained=False)
    online = ReplayBuffer(4)
    offline = ReplayBuffer(4)
    for _ in range(2):
        item = transition(observation(), np.array([0, 0, 0, 1], dtype=np.float32),
                          0.0, observation(), False, {})
        online.add(item)
        offline.add(item)
    stats = agent.update(lambda: batch(online, offline, 2, device))
    assert all(np.isfinite(value) for value in stats.values())
    action = agent.act(online.items[0].observation).numpy()
    assert action.shape == (4,)
    assert (np.abs(action[:3]) <= 1).all()
    assert action[3] in (0, 1, 2)


def test_actor_replica_syncs_learner_weights():
    torch.set_num_threads(1)
    device = torch.device("cpu")
    config = HILSerlConfig(hidden_dim=32, latent_dim=16)
    learner = HILSerl(config, device, pretrained=False)
    actor = ActorReplica(config, device, pretrained=False)
    actor.load_weights(learner.actor_weights())
    expected = learner.act(learner_observation(), deterministic=True)
    actual = actor.act(learner_observation(), deterministic=True)
    torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        learner.actor.network[-1].bias[:3].add_(1)
    actor.load_weights(learner.actor_weights())
    updated = actor.act(learner_observation(), deterministic=True)
    torch.testing.assert_close(updated, learner.act(learner_observation(), deterministic=True))
    assert not torch.equal(updated[:3], actual[:3])


def test_pretrained_weight_sync_omits_frozen_backbone():
    torch.set_num_threads(1)
    device = torch.device("cpu")
    config = HILSerlConfig(hidden_dim=32, latent_dim=16)
    learner = HILSerl(config, device, pretrained=False)
    actor = ActorReplica(config, device, pretrained=False)
    learner.encoder.pretrained = True
    actor.encoder.pretrained = True
    backbone_before = actor.encoder.backbone.conv1.weight.detach().clone()
    with torch.no_grad():
        learner.encoder.image_projection.bias.add_(1)
    weights = learner.actor_weights()
    assert not any(key.startswith("backbone.") for key in weights["encoder"])
    actor.load_weights(weights)
    torch.testing.assert_close(actor.encoder.backbone.conv1.weight, backbone_before)
    torch.testing.assert_close(actor.encoder.image_projection.bias, learner.encoder.image_projection.bias)
    with pytest.raises(ValueError, match="incomplete encoder weights"):
        actor.load_weights({**weights, "encoder": {}})


def learner_observation():
    raw = observation()
    return {"state": raw["agent_pos"], **raw["pixels"]}


def test_pick_cube_always_enables_viewer_and_intervention(monkeypatch):
    calls = []
    viewer_calls = []
    monkeypatch.setattr("robot_rl.keyboard.ManipulationKeyboard.start", lambda self: None)
    monkeypatch.setattr(pick_cube, "KeyboardSafeViewer",
                        lambda env: viewer_calls.append(env) or env)

    def make_stub_environment(task_id, **kwargs):
        calls.append((task_id, kwargs))
        return pick_cube.gym.Env()

    monkeypatch.setattr(pick_cube.gym, "make", make_stub_environment)
    for gamepad, task_id in ((False, "gym_hil/PandaPickCubeKeyboard-v0"),
                             (True, "gym_hil/PandaPickCubeGamepad-v0")):
        environment = pick_cube.make_env(gamepad=gamepad)
        assert calls[-1][0] == task_id
        assert calls[-1][1]["use_viewer"] is gamepad
        assert calls[-1][1]["use_inputs_control"] is gamepad
        assert calls[-1][1]["use_gamepad"] is gamepad
        assert calls[-1][1]["auto_reset"] is False
        assert calls[-1][1]["max_episode_steps"] == 500
        if not gamepad:
            assert environment.controller is not None
        environment.close()
    assert len(viewer_calls) == 1


def test_pick_cube_episode_limit_is_not_capped_by_gym_hil_default(monkeypatch):
    monkeypatch.setattr("robot_rl.keyboard.ManipulationKeyboard.start", lambda self: None)
    monkeypatch.setattr(pick_cube, "KeyboardSafeViewer", lambda env: env)

    class EmptyTask(pick_cube.gym.Env):
        def reset(self, seed=None, **kwargs):
            return {}, {}

        def step(self, action):
            return {}, 0.0, False, False, {}

    def make_stub_environment(task_id, **kwargs):
        return pick_cube.gym.wrappers.TimeLimit(EmptyTask(), kwargs["max_episode_steps"])

    monkeypatch.setattr(pick_cube.gym, "make", make_stub_environment)
    environment = pick_cube.make_env(max_episode_steps=250)
    environment.reset()
    for _ in range(249):
        assert not environment.step(None)[3]
    assert environment.step(None)[3]
    environment.close()
