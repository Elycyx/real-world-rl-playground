"""PyTorch hybrid SAC + double-DQN gripper with RLPD critic updates."""

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, ResNet


@dataclass
class HILSerlConfig:
    discount: float = 0.97
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    initial_temperature: float = 0.01
    utd_ratio: int = 2
    grad_clip: float = 10.0
    latent_dim: int = 64
    hidden_dim: int = 256


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


RESNET10_REVISION = "60e4365989619db6365ba92b1169d5dcc143d165"


class VisualEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, pretrained: bool):
        super().__init__()
        self.pretrained = pretrained
        if pretrained:
            try:
                from transformers import AutoModel
            except ImportError as error:
                raise ImportError("Install the pretrained extra: pip install -e '.[pretrained]'") from error
            try:
                self.backbone = AutoModel.from_pretrained("lerobot/resnet10", revision=RESNET10_REVISION,
                                                           trust_remote_code=True)
            except OSError as error:
                if "Unknown scheme for proxy URL" in str(error) and "socks://" in str(error):
                    raise RuntimeError(
                        "Unsupported socks:// proxy while loading lerobot/resnet10. "
                        "If HTTP_PROXY and HTTPS_PROXY are already configured, retry with "
                        "'env -u ALL_PROXY -u all_proxy uv run --locked --extra pretrained robot-rl train ...'. "
                        "For a SOCKS-only proxy, use socks5:// and install httpx[socks]."
                    ) from error
                raise
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()
        else:
            self.backbone = ResNet(BasicBlock, [1, 1, 1, 1])
            self.backbone.fc = nn.Identity()
        self.image_projection = nn.Linear(1024, hidden_dim)
        self.state_projection = nn.Linear(18, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, latent_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.register_buffer("state_min", torch.tensor([
            -0.18890867, -0.13319267, -0.15566483, -3.04198435, -0.05300621,
            2.52864378, 0.426834, -100, -100, -100, -100, -100, -100, -100,
            0, 0.2, -0.3, 0,
        ]))
        self.register_buffer("state_max", torch.tensor([
            0.1956124, 0.5096723, 0.05482981, -2.01792539, 0.04425749,
            2.90565838, 1.07170398, 100, 100, 100, 100, 100, 100, 100,
            255, 0.6, 0.3, 0.5,
        ]))

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        images = []
        for camera in ("front", "wrist"):
            image = observation[camera].permute(0, 3, 1, 2).float() / 255.0
            if self.pretrained:
                with torch.no_grad():
                    features = self.backbone((image - self.mean) / self.std).last_hidden_state
                    features = features.mean(dim=(-2, -1))
            else:
                features = self.backbone((image - self.mean) / self.std)
            images.append(features)
        visual = F.relu(self.image_projection(torch.cat(images, dim=-1)))
        normalized_state = 2 * ((observation["state"].float() - self.state_min) /
                                (self.state_max - self.state_min)) - 1
        state = F.relu(self.state_projection(normalized_state))
        return F.relu(self.fusion(torch.cat((visual, state), dim=-1)))


class Actor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.network = mlp(latent_dim, 6, hidden_dim)

    def forward(self, features: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(features).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        action = pre_tanh.tanh()
        log_prob = (distribution.log_prob(pre_tanh) -
                    2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(-1)
        return action, log_prob


@torch.no_grad()
def select_action(encoder: VisualEncoder, actor: Actor, gripper: nn.Module,
                  device: torch.device, observation: dict, deterministic: bool = False) -> torch.Tensor:
    tensors = {key: torch.as_tensor(value, device=device).unsqueeze(0)
               for key, value in observation.items()}
    was_training = encoder.training
    encoder.eval()
    try:
        features = encoder(tensors)
    finally:
        encoder.train(was_training)
    continuous, _ = actor(features, deterministic=deterministic)
    discrete = gripper(features).argmax(dim=-1, keepdim=True)
    return torch.cat((continuous, discrete.float()), dim=-1).squeeze(0).cpu()


class ActorReplica:
    def __init__(self, config: HILSerlConfig, device: torch.device, pretrained: bool = True):
        self.device = device
        self.encoder = VisualEncoder(config.latent_dim, config.hidden_dim, pretrained).to(device)
        self.actor = Actor(config.latent_dim, config.hidden_dim).to(device)
        self.gripper = mlp(config.latent_dim, 3, config.hidden_dim).to(device)
        self.encoder.eval()
        self.actor.eval()
        self.gripper.eval()

    def load_weights(self, weights: dict) -> None:
        for name in ("encoder", "actor", "gripper"):
            module = getattr(self, name)
            if name == "encoder" and self.encoder.pretrained:
                expected = {key for key in module.state_dict() if not key.startswith("backbone.")}
                if set(weights[name]) != expected:
                    raise ValueError("Pretrained actor received incomplete encoder weights")
            module.load_state_dict({key: torch.as_tensor(value) for key, value in weights[name].items()},
                                   strict=not (name == "encoder" and self.encoder.pretrained))

    def act(self, observation: dict, deterministic: bool = False) -> torch.Tensor:
        return select_action(self.encoder, self.actor, self.gripper, self.device, observation, deterministic)


class HILSerl:
    def __init__(self, config: HILSerlConfig, device: torch.device, pretrained: bool = True):
        self.config = config
        self.device = device
        self.encoder = VisualEncoder(config.latent_dim, config.hidden_dim, pretrained).to(device)
        self.actor = Actor(config.latent_dim, config.hidden_dim).to(device)
        self.critics = nn.ModuleList([mlp(config.latent_dim + 3, 1, config.hidden_dim).to(device)
                                      for _ in range(2)])
        self.target_critics = copy.deepcopy(self.critics)
        self.gripper = mlp(config.latent_dim, 3, config.hidden_dim).to(device)
        self.target_gripper = copy.deepcopy(self.gripper)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config.initial_temperature), device=device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        critic_parameters = list(self.critics.parameters()) + list(self.encoder.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.critic_lr)
        self.gripper_optimizer = torch.optim.Adam(self.gripper.parameters(), lr=config.critic_lr)
        self.temperature_optimizer = torch.optim.Adam([self.log_alpha], lr=config.temperature_lr)

    @torch.no_grad()
    def act(self, observation: dict, deterministic: bool = False) -> torch.Tensor:
        return select_action(self.encoder, self.actor, self.gripper, self.device, observation, deterministic)

    def actor_weights(self) -> dict:
        return {name: {key: value.detach().cpu().numpy().copy()
                       for key, value in getattr(self, name).state_dict().items()
                       if not (name == "encoder" and self.encoder.pretrained and key.startswith("backbone."))}
                for name in ("encoder", "actor", "gripper")}

    def _soft_update(self) -> None:
        for source, target in ((self.critics, self.target_critics), (self.gripper, self.target_gripper)):
            for parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.data.lerp_(parameter.data, self.config.tau)

    def update(self, sample_batch) -> dict[str, float]:
        stats = {}
        for update_index in range(self.config.utd_ratio):
            batch = sample_batch()
            features = self.encoder(batch["obs"])
            with torch.no_grad():
                next_features = self.encoder(batch["next_obs"])
                next_action, next_log_prob = self.actor(next_features)
                next_q = torch.stack([critic(torch.cat((next_features, next_action), dim=-1)).squeeze(-1)
                                      for critic in self.target_critics]).min(dim=0).values
                target_q = batch["reward"] + self.config.discount * (1 - batch["done"]) * (
                    next_q - self.log_alpha.exp() * next_log_prob)
                next_gripper = self.gripper(next_features).argmax(dim=-1, keepdim=True)
                target_gripper = batch["reward"] + batch["penalty"] + self.config.discount * (
                    1 - batch["done"]) * self.target_gripper(next_features).gather(1, next_gripper).squeeze(1)

            critic_input = torch.cat((features, batch["action"][:, :3]), dim=-1)
            critic_loss = sum(F.mse_loss(critic(critic_input).squeeze(-1), target_q)
                              for critic in self.critics)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(self.critics.parameters()) + list(self.encoder.parameters()),
                                     self.config.grad_clip)
            self.critic_optimizer.step()

            gripper_features = self.encoder(batch["obs"]).detach()
            predicted_gripper = self.gripper(gripper_features).gather(
                1, batch["action"][:, 3].round().long().unsqueeze(1)).squeeze(1)
            gripper_loss = F.mse_loss(predicted_gripper, target_gripper)
            self.gripper_optimizer.zero_grad()
            gripper_loss.backward()
            nn.utils.clip_grad_norm_(self.gripper.parameters(), self.config.grad_clip)
            self.gripper_optimizer.step()
            self._soft_update()

            if update_index == self.config.utd_ratio - 1:
                actor_features = self.encoder(batch["obs"]).detach()
                actions, log_prob = self.actor(actor_features)
                for parameter in self.critics.parameters():
                    parameter.requires_grad_(False)
                q_value = torch.stack([critic(torch.cat((actor_features, actions), dim=-1)).squeeze(-1)
                                       for critic in self.critics]).min(dim=0).values
                actor_loss = (self.log_alpha.exp().detach() * log_prob - q_value).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
                self.actor_optimizer.step()
                for parameter in self.critics.parameters():
                    parameter.requires_grad_(True)
                temperature_loss = -(self.log_alpha.exp() * (log_prob.detach() - 1.5)).mean()
                self.temperature_optimizer.zero_grad()
                temperature_loss.backward()
                self.temperature_optimizer.step()
                stats = {"critic": critic_loss.item(), "gripper": gripper_loss.item(),
                         "actor": actor_loss.item(), "alpha": self.log_alpha.exp().item()}
        return stats

    def state_dict(self) -> dict:
        return {key: getattr(self, key).state_dict() for key in
                ("encoder", "actor", "critics", "target_critics", "gripper", "target_gripper",
                 "actor_optimizer", "critic_optimizer", "gripper_optimizer", "temperature_optimizer")} | {
                     "log_alpha": self.log_alpha.detach().clone()}

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            if key == "log_alpha":
                self.log_alpha.data.copy_(value)
            else:
                getattr(self, key).load_state_dict(value)
