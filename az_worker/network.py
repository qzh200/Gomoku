from __future__ import annotations

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
    def __init__(self, channels: int = 64, blocks: int = 6, value_hidden: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.value_hidden = value_hidden
        self.stem = nn.Conv2d(PLANE_COUNT, channels, 3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(channels)
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_norm = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ACTION_COUNT, ACTION_COUNT)

        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_norm = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(ACTION_COUNT, value_hidden)
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
    planes: list[list[list[float]]],
    masks: list[list[float]],
    device: torch.device,
) -> tuple[list[list[float]], list[float]]:
    model.eval()
    with torch.no_grad():
        batch = torch.tensor(planes, dtype=torch.float32, device=device)
        mask = torch.tensor(masks, dtype=torch.float32, device=device)
        logits, values = model(batch)


        logits = logits.masked_fill(mask == 0, -1e9)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.cpu().tolist(), values.cpu().tolist()


def train_batch(
    model: GomokuNet,
    optimizer: torch.optim.Optimizer,
    planes: list[list[list[float]]],
    policies: list[list[float]],
    values: list[float],
    legal: list[list[float]],
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    batch = torch.tensor(planes, dtype=torch.float32, device=device)
    target_policy = torch.tensor(policies, dtype=torch.float32, device=device)
    target_value = torch.tensor(values, dtype=torch.float32, device=device)
    mask = torch.tensor(legal, dtype=torch.float32, device=device)
    return train_tensors(model, optimizer, batch, target_policy, target_value, mask)


def train_tensors(
    model: GomokuNet,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    mask: torch.Tensor,
    use_amp: bool = True,
) -> tuple[float, float, float]:
    model.train()

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp and batch.is_cuda):
        logits, predicted = model(batch)
        logits = logits.float()
        predicted = predicted.float()

    logits = logits.masked_fill(mask == 0, -1e9)
    log_probabilities = torch.log_softmax(logits, dim=1)


    target_policy = target_policy * mask
    total = target_policy.sum(dim=1, keepdim=True)
    target_policy = target_policy / torch.clamp(total, min=1e-8)
    policy_loss = -(target_policy * log_probabilities).sum(dim=1).mean()
    value_loss = ((predicted - target_value) ** 2).mean()
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def save(model: GomokuNet, path: str) -> None:
    torch.save(
        {
            "state": model.state_dict(),
            "channels": model.channels,
            "blocks": model.blocks,
            "value_hidden": model.value_hidden,
        },
        path,
    )


def _count_blocks(state: dict) -> int:
    names = {name for name in state if name.startswith("tower.") and name.endswith("conv1.weight")}
    return len(names)


def load_model(path: str, device: torch.device) -> GomokuNet:
    payload = torch.load(path, map_location=device)
    state = payload["state"]
    channels = int(payload.get("channels", state["stem.weight"].shape[0]))
    blocks = int(payload.get("blocks", _count_blocks(state)))
    value_hidden = int(payload.get("value_hidden", state["value_fc1.weight"].shape[0]))
    model = GomokuNet(channels=channels, blocks=blocks, value_hidden=value_hidden)
    model.load_state_dict(state)
    return model.to(device)
