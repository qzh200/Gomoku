# 训练手册（az_worker）

三步，顺序不能换。全部在 `E:/Gomoku` 下执行。

## 0. 前提

```sh
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

必须输出 `True`。**用 `.venv/Scripts/python.exe`**，系统 Python 没有 torch。

---

## 1. 预热（必做，约 15–20 分钟）

传统引擎当老师，把棋型知识灌进网络。**跳过这步，第 2 步什么也学不到**。

```sh
.venv/Scripts/python.exe -m az_worker.warmstart \
  --games 300 --time 0.1 --depth 4 --width 12 \
  --steps 2000 --teacher-moves 4 --decay 1.5 \
  --value-target outcome \
  --model models/az/warmstart.pt
```

看两个数：

| 输出 | 含义 | 目标 |
| --- | --- | --- |
| `验证首选一致率` | 网络首选**正好是**老师第一手的比例 | ≥ 40% |
| `价值范围` | 价值的取值跨度 | 覆盖 −1 到 1 |

一致率低于 30% 就加大 `--steps`（3000）或 `--games`（500）。**不要**用
`--value-target eval`——两个阶段教不同的东西，切换时价值头的标定会整个坏掉
（实测验证价值损失从 0.25 跳到 2.5）。

产物：`models/az/warmstart.pt`

---

## 2. 自对弈（主要训练时间）

```sh
.venv/Scripts/python.exe -m az_worker.train \
  --init models/az/warmstart.pt \
  --model models/az/current.pt \
  --buffer-file models/az/current.buffer \
  --games 20 --iterations 200 \
  --simulations 128 --train-steps 400 --batch-size 128 \
  --lr 1e-4 --device cuda
```

`--init` 和 `--model` 分开：前者是起点，后者是产物，预热权重不会被覆盖。

每轮一行输出，**只看「验证」两个数**：

```
第 7 轮: 胜负 11:9，缓冲区 1258，训练损失 2.53，验证策略 2.33 验证价值 1.19，用时 80 秒
```

- `验证策略` 应当缓慢下降。**如果它跟着训练损失一起下降但验证不动，就是过拟合**，
  降低 `--lr`（1e-4 → 3e-5）或加大 `--games`。
- `验证价值` 会在 1.0 附近徘徊，这是正常的：整局结果本来就难预测。别指望它降到 0。
- `缓冲区` 不涨就是自对弈没在产数据，检查 `--games`。

一轮约 70–90 秒（20 局 × 128 模拟）。200 轮约 4–5 小时。

### 中断与续跑

每轮结束都会写 `current.pt` 和 `current.buffer`，续跑：

```sh
.venv/Scripts/python.exe -m az_worker.train \
  --model models/az/current.pt --buffer-file models/az/current.buffer --resume \
  --games 20 --iterations 200 --simulations 128 \
  --train-steps 400 --batch-size 128 --lr 1e-4 --device cuda
```

`--resume` 会读回缓冲区；不加 `--init` 就是从 `current.pt` 继续。

### 建议的中途检查

每 20–30 轮跑一次擂台赛，看棋力而不是看损失：

```sh
.venv/Scripts/python.exe -m az_worker.arena \
  --games 20 --simulations 200 --device cuda --model models/az/current.pt
```

---

## 3. 擂台赛（验收）

```sh
.venv/Scripts/python.exe -m az_worker.arena --games 30 --simulations 200 --device cuda
```

对局的是 `classical_worker`（深度 4 / 0.5 秒，`arena.py` 的默认值），双方换先手。

**实话：这个引擎目前打不过传统引擎。**把网络整个换成传统引擎自己的策略与价值
（oracle），AlphaZero 的搜索仍然以 1:5 输——差距在搜索结构，不在网络。训练能让
它更强，但别指望超过传统引擎。

擂台赛的价值是**相对自己的基线**：第 1 步之后的成绩是基线，训练后应该更高。

---

## 4. 对局

```sh
go run ./cmd/app                          # 先起 MyGo，它写令牌
.venv/Scripts/python.exe -m az_worker     # 再起 worker
```

`configs/engines.json` 里要有 `az` 这个引擎（`configs/engines.example.json` 是模板）。

---

## 参数速查

| 参数 | 建议 | 说明 |
| --- | --- | --- |
| `--games` | 20 | 每轮自对弈局数。大一点能防过拟合 |
| `--iterations` | 200 | 轮数 |
| `--simulations` | 128 | 每手模拟次数。256 更准但慢一倍 |
| `--train-steps` | 400 | 每轮训练步数 |
| `--batch-size` | 128 | 显存够可以到 256 |
| `--lr` | 1e-4 | 学习率。觉得过拟合就降到 3e-5 |
| `--buffer` | 60000 | 缓冲区容量 |
| `--candidate-radius` | 1 | **别改成 2**，分布会变平 |
| `--device` | cuda | |

## 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 验证策略一直 4.2 不动 | 没做预热 | 回到第 1 步 |
| 训练损失降、验证不降 | 过拟合 | 降 `--lr` 或加 `--games` |
| 验证价值从 0.2 跳到 2.5 | 预热用了 `--value-target eval` | 改用 `outcome` 重做预热 |
| 一致率只有 20% | 数据太少 | 加 `--games` / `--steps` |
| `FileNotFoundError: torch` | 用了系统 Python | 改用 `.venv/Scripts/python.exe` |
| CUDA 内存不足 | 批太大 | `--batch-size 64` |
