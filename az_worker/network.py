from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .board import BOARD_SIZE
from .features import PLANE_COUNT


ACTION_COUNT = BOARD_SIZE * BOARD_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + x)


class GomokuNet(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        blocks: int = 6,
        value_hidden: int = 64,
        policy_channels: int = 2,
        value_channels: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.value_hidden = value_hidden
        self.policy_channels = policy_channels
        self.value_channels = value_channels
        self.stem = nn.Conv2d(PLANE_COUNT, channels, 3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(channels)
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_norm = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * ACTION_COUNT, ACTION_COUNT)

        self.value_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.value_norm = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * ACTION_COUNT, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.stem_norm(self.stem(x)))
        out = self.tower(out)

        policy = F.relu(self.policy_norm(self.policy_conv(out)))
        policy = self.policy_fc(policy.flatten(1))

        value = F.relu(self.value_norm(self.value_conv(out)))
        value = F.relu(self.value_fc1(value.flatten(1)))
        value = torch.tanh(self.value_fc2(value))
        return policy, value.view(-1)


def evaluate(
    model: GomokuNet,
    planes: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(np.ascontiguousarray(planes)).to(device)
        mask = torch.from_numpy(np.ascontiguousarray(masks)).to(device)
        logits, values = model(batch)
        logits = logits.masked_fill(mask == 0, -1e9)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.cpu().numpy(), values.cpu().numpy()


def train_batch(
    model: GomokuNet,
    optimizer: torch.optim.Optimizer,
    planes: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float, float]:
    model.train()

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=planes.is_cuda):
        logits, predicted = model(planes)
        logits = logits.float()
        predicted = predicted.float()

    logits = logits.masked_fill(mask == 0, -1e9)
    log_probabilities = torch.log_softmax(logits, dim=1)
    target_policy = target_policy * mask
    target_policy = target_policy / torch.clamp(target_policy.sum(dim=1, keepdim=True), min=1e-8)
    policy_loss = -(target_policy * log_probabilities).sum(dim=1).mean()
    value_loss = ((predicted - target_value) ** 2).mean()
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def save(model: GomokuNet, path: str, meta: dict | None = None) -> None:
    torch.save(
        {
            "state": model.state_dict(),
            "channels": model.channels,
            "blocks": model.blocks,
            "value_hidden": model.value_hidden,
            "policy_channels": model.policy_channels,
            "value_channels": model.value_channels,
            "planes": PLANE_COUNT,
            "meta": dict(meta or {}),
        },
        path,
    )


def read_meta(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("meta") or {})


def _count_blocks(state: dict) -> int:
    names = {name for name in state if name.startswith("tower.") and name.endswith("conv1.weight")}
    return len(names)


def load_model(path: str, device: torch.device) -> GomokuNet:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["state"]
    planes = int(payload.get("planes", state["stem.weight"].shape[1]))
    if planes != PLANE_COUNT:
        raise ValueError(
            f"{path} 是 {planes} 平面权重，现在是 {PLANE_COUNT} 平面，旧权重不能再用，要从随机网络重训"
        )
    model = GomokuNet(
        channels=int(payload.get("channels", state["stem.weight"].shape[0])),
        blocks=int(payload.get("blocks", _count_blocks(state))),
        value_hidden=int(payload.get("value_hidden", state["value_fc1.weight"].shape[0])),
        policy_channels=int(payload.get("policy_channels", state["policy_conv.weight"].shape[0])),
        value_channels=int(payload.get("value_channels", state["value_conv.weight"].shape[0])),
    )
    model.load_state_dict(state)
    return model.to(device)
