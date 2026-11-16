from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .features import CELL_PERMUTATIONS, PLANE_COUNT, as_planes
from .selfplay import Example


PERMUTATION_TABLE = torch.from_numpy(np.stack(CELL_PERMUTATIONS).astype(np.int64))


class TensorDataset:
    """训练数据全部放在显卡上，采样和对称增广也在显卡里做。

    增广用同一个格子置换表：每个平面摊平成 225 格后，
    八种对称变换都只是对格子的一个置换，策略和合法掩码同理。
    """

    def __init__(self, device: torch.device, capacity: int = 60_000) -> None:
        self.device = device
        self.capacity = max(1, capacity)
        self.planes = torch.zeros(
            (self.capacity, PLANE_COUNT, 225), dtype=torch.float16, device=device
        )
        self.policy = torch.zeros((self.capacity, 225), dtype=torch.float32, device=device)
        self.legal = torch.zeros((self.capacity, 225), dtype=torch.bool, device=device)
        self.value = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self.count = 0
        self.perm = PERMUTATION_TABLE.to(device)

    def __len__(self) -> int:
        return self.count

    def add(self, example: Example) -> None:
        self.extend([example])

    def extend(self, examples: list[Example]) -> None:
        if not examples:
            return
        if len(examples) >= self.capacity:
            examples = examples[-self.capacity :]
            self.count = 0
        if self.count + len(examples) > self.capacity:
            overflow = self.count + len(examples) - self.capacity
            self.planes[: self.count - overflow] = self.planes[overflow : self.count].clone()
            self.policy[: self.count - overflow] = self.policy[overflow : self.count].clone()
            self.legal[: self.count - overflow] = self.legal[overflow : self.count].clone()
            self.value[: self.count - overflow] = self.value[overflow : self.count].clone()
            self.count -= overflow
        start = self.count
        end = start + len(examples)
        planes = torch.from_numpy(
            np.stack([as_planes(example.planes) for example in examples]).astype(np.float16)
        ).reshape(len(examples), PLANE_COUNT, 225)
        policy = torch.from_numpy(
            np.stack([example.policy for example in examples]).astype(np.float32)
        )
        legal = torch.from_numpy(
            np.stack([example.legal for example in examples]) > 0.5
        )
        value = torch.from_numpy(
            np.array([example.value for example in examples], dtype=np.float32)
        )
        self.planes[start:end] = planes.to(self.device, non_blocking=True)
        self.policy[start:end] = policy.to(self.device, non_blocking=True)
        self.legal[start:end] = legal.to(self.device, non_blocking=True)
        self.value[start:end] = value.to(self.device, non_blocking=True)
        self.count = end

    def merge(self, other: "TensorDataset") -> None:
        """把另一个数据集并进来，样本留在显卡上。"""
        count = len(other)
        if count == 0:
            return
        if count >= self.capacity:
            self.planes[:] = other.planes[: self.capacity]
            self.policy[:] = other.policy[: self.capacity]
            self.legal[:] = other.legal[: self.capacity]
            self.value[:] = other.value[: self.capacity]
            self.count = self.capacity
            return
        start = self.count
        end = start + count
        if end > self.capacity:
            keep = self.capacity - start
            self.planes[start : self.capacity] = other.planes[:keep]
            self.policy[start : self.capacity] = other.policy[:keep]
            self.legal[start : self.capacity] = other.legal[:keep]
            self.value[start : self.capacity] = other.value[:keep]
            self.count = self.capacity
        else:
            self.planes[start:end] = other.planes[:count]
            self.policy[start:end] = other.policy[:count]
            self.legal[start:end] = other.legal[:count]
            self.value[start:end] = other.value[:count]
            self.count = end

    def sample(
        self,
        batch_size: int,
        rng: random.Random | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        picker = rng or random.Random(0)
        limit = self.count if self.count < self.capacity else self.capacity
        generator = torch.Generator().manual_seed(picker.randrange(1 << 30))
        indexes = torch.randint(
            limit,
            (batch_size,),
            generator=generator,
        ).to(self.device)
        planes = self.planes[indexes].float()
        policy = self.policy[indexes]
        legal = self.legal[indexes].float()
        value = self.value[indexes]
        if augment:
            kinds = torch.randint(
                len(self.perm),
                (batch_size,),
                generator=generator,
            ).to(self.device)
            permutation = self.perm[kinds]
            planes = torch.gather(
                planes, 2, permutation.unsqueeze(1).expand(-1, PLANE_COUNT, -1)
            )
            policy = torch.gather(policy, 1, permutation)
            legal = torch.gather(legal, 1, permutation)
        return planes.reshape(-1, PLANE_COUNT, 15, 15), policy, legal, value

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "planes": self.planes[: self.count].cpu(),
                "policy": self.policy[: self.count].cpu(),
                "legal": self.legal[: self.count].cpu(),
                "value": self.value[: self.count].cpu(),
            },
            str(target),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device,
        capacity: int | None = None,
    ) -> "TensorDataset":
        source = Path(path)
        if not source.is_file():
            return cls(device, capacity or 60_000)
        payload = torch.load(str(source), map_location="cpu", weights_only=False)
        if isinstance(payload, list):
            dataset = cls(device, capacity or max(len(payload), 1))
            dataset.extend(payload)
            return dataset
        count = payload["planes"].shape[0]
        if payload["planes"].shape[1] != PLANE_COUNT:
            raise ValueError(
                f"{source} 里的平面数是 {payload['planes'].shape[1]}，"
                f"现在是 {PLANE_COUNT}，这份数据过期了，要重新采集"
            )
        dataset = cls(device, capacity or max(count, 1))
        dataset.planes[:count] = payload["planes"].to(device)
        dataset.policy[:count] = payload["policy"].to(device)
        dataset.legal[:count] = payload["legal"].to(device)
        dataset.value[:count] = payload["value"].to(device)
        dataset.count = count
        return dataset
