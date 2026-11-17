# AlphaZero 引擎（az_worker）

`az_worker` 是策略/价值网络 + PUCT MCTS，训练方式是“传统引擎预热 + 自对弈迭代”。推理时不需要外部服务，模型在本地 GPU/CPU 上运行。

## 1. 组件

```mermaid
flowchart LR
  B["board.py\n15×15 棋盘"] --> F["features.py\n7 平面编码"]
  T["threats.py\nbitboard 威胁"] --> F
  F --> N["network.py\n残差网络"]
  N --> M["mcts.py\nPUCT + 虚拟损失"]
  M --> E["engine.py\nchoose_move"]
  E --> W["aiworker.Worker"]
  C["classical_worker\n教师"] --> WS["warmstart.py"]
  WS --> N
  SP["selfplay.py"] --> TR["train.py"]
  N --> SP
  TR --> N
  M --> AR["arena.py"]
  C --> AR
```

| 文件 | 职责 |
| --- | --- |
| `board.py` | 棋盘、落子、成五判定、候选点 |
| `threats.py` | 位棋盘上的成五点/活四点掩码 |
| `features.py` | 7 平面编码、合法点掩码、8 种对称变换 |
| `network.py` | 网络结构、推理、训练步、保存/加载 |
| `mcts.py` | 节点、PUCT 选择、批量叶评估、Dirichlet 噪声 |
| `engine.py` | Worker 适配、强制手、时间/模拟数预算 |
| `selfplay.py` | 自对弈、样本、GPU 训练辅助、评估 |
| `dataset.py` | GPU 张量回放缓冲、对称增强 |
| `teacher.py` | 用传统引擎收集教师样本 |
| `warmstart.py` | 预热训练入口 |
| `train.py` | 自对弈训练入口 |
| `arena.py` | 与传统引擎的擂台赛 |
| `config.py` / `__main__.py` | 环境参数与 Worker 启动 |

## 2. 棋盘与候选点

`Board.grid` 是 15×15 列表，`_stones` 维护 `(row, col, player)` 列表。`play` 返回是否成五。

`candidates(radius=2)`：

- 空盘返回天元 `(7,7)`；
- 否则返回所有距已有棋子 ≤radius 的空格，排序后输出。

MCTS 内部实际使用 `radius=1` 的候选集（`DEFAULT_CANDIDATE_RADIUS = 1`），因为 15×15 中盘中段 radius 2 会产生 60+ 个点，模拟数被摊薄。

## 3. 七平面特征

`features.encode(board, player, last_move)` 返回 `(7, 15, 15) float32`：

| 索引 | 平面 | 含义 |
| --- | --- | --- |
| 0 | `PLANE_SELF` | 当前行动方棋子 |
| 1 | `PLANE_OTHER` | 对方棋子 |
| 2 | `PLANE_LAST` | 最后一手 |
| 3 | `PLANE_SELF_FIVE` | 我方落子即胜的空点 |
| 4 | `PLANE_OTHER_FIVE` | 对方落子即胜的空点 |
| 5 | `PLANE_SELF_OPEN_FOUR` | 我方落子成活四的空点 |
| 6 | `PLANE_OTHER_OPEN_FOUR` | 对方落子成活四的空点 |

后四个平面**相对于当前行动方**，因此会随回合交换“我方/对方”的含义，和平面 0/1 一致。

加入威胁平面的原因：三平面网络只能看到“子在哪里”，看不到“对方下一手在哪里成五”。实测在对方已有冲四的 441 个局面上，正确堵点比例从 12.5% 提升到 47.4%。

### 3.1 bitboard 表示

`threats.py`：

```text
STRIDE = BOARD_SIZE + 5 = 20
TOTAL_BITS = STRIDE * 15 = 300
bit(row, col) = 1 << (row * STRIDE + col)
```

每行留 5 个填充位，使四个方向都变成固定位移：

```text
DIRECTIONS = (1, 20, 21, 19)   # 横、竖、主对角、副对角
```

填充位保证水平移位不会从上一行末尾绕到下一行开头。`VALID_MASK` 只包含真实 225 格。

### 3.2 成五点与活四点

对每个方向 `shift`：

```python
gaps = [empty >> (j*shift) for j in range(6)]   # 6 个连续格子的空位掩码
ends = gaps[0] & gaps[5]                        # 六格窗口两端都空
line = [stones >> (j*shift) for j in range(5)]  # 五格窗口内己方子
```

`_products` 计算前缀/后缀 AND：

```text
prefix[i+1] = prefix[i] & line[i]
suffix[i]   = suffix[i+1] & line[i]
```

- **成五点**：五格窗口中恰好一个空位，其余四格为己方：

```text
five |= (gaps[hole] & prefix[hole] & suffix[hole+1]) << (hole*shift)
```

- **活四点**：六格窗口两端为空，中间四格为己方，窗口内剩余那个空位就是“落子成活四”的点。对中间四格再做前缀/后缀 AND：

```text
middle_prefix, middle_suffix = products(line[1:5])
others = middle_prefix[hole] & middle_suffix[hole+1]
four |= (ends & gaps[hole+1] & others) << ((hole+1)*shift)
```

所有方向的结果取或，最后 `& empty` 去掉已有棋子的位。

一次计算 4 个方向 × 2 方，都是整数位运算，没有逐格 Python 循环。`encode` 从 5.5µs 增加到约 34µs，这是 4 个额外平面的代价。

### 3.3 对称

`features._transforms` 定义 8 种棋盘对称（恒等、旋转 90/180/270、水平翻转及其旋转）。`CELL_PERMUTATIONS` 是对应的**逆向**索引，用于 GPU `gather`。

`transform_planes(encode(board)) == encode(permuted_board)` 对 8 种对称都成立；威胁平面由同一 bitboard 几何计算，因此自动满足。

## 4. 网络

`network.GomokuNet(channels=64, blocks=6, value_hidden=64)`：

```text
输入  (B, 7, 15, 15)
stem  Conv2d(7→64, 3×3, pad=1, bias=False) + BN + ReLU
tower 6 × ResidualBlock(64)
       └ Conv3×3 + BN + ReLU → Conv3×3 + BN → +x → ReLU

policy head:
  Conv2d(64→2, 1×1) + BN + ReLU
  flatten → Linear(2*225 → 225) → logits

value head:
  Conv2d(64→1, 1×1) + BN + ReLU
  flatten → Linear(225 → 64) + ReLU → Linear(64 → 1) → tanh
```

推理：

- logits 在非法点（`legal_mask == 0`）上填 `-1e9`；
- `softmax` 得到先验；
- value 在 `[-1, 1]`。

训练：

```text
policy_loss = -sum(target_policy * log_softmax(masked_logits))
value_loss  = MSE(predicted, target_value)
loss        = policy_loss + value_loss
```

目标策略先乘合法掩码并重新归一化。`train_tensors` 在 CUDA 上用 bf16 autocast，loss 仍以 fp32 计算。

`load_model` 支持旧三平面权重：`_expand_stem` 把 `stem.weight` 的输入通道零填充到 7，旧模型行为不变。

## 5. MCTS

### 5.1 节点

```python
class Node:
    children: dict[int, Node]   # cell = row*15+col
    visits, value_sum, virtual_visits
    prior, player
    terminal_value
    expanded
    last_move
    _board, _parent, _move
```

`Node.board` 是惰性属性：只有访问时才从父节点复制棋盘并落子。旧实现每个子节点都复制整盘，400 次模拟产生 26,379 次 `copy`，占 44% 运行时间；惰性化后 400 次模拟从 782ms 降到 359ms。

### 5.2 选择

```text
quality = -child.value_sum / child.visits      # 子节点价值是父方视角的相反数
exploration = c_puct * prior * sqrt(parent.visits) / (1 + child.visits + child.virtual_visits)
score = quality + exploration
```

`c_puct = 1.5`。未访问子节点 `quality = 0`，靠 prior 和探索项决定。

`virtual_visits` 实现虚拟损失：一个叶节点正在等 GPU 评估时，其他下降路径会看到额外的访问计数，避免重复选同一个叶子。

### 5.3 批量评估

`search` 主循环：

1. `_descend` 沿已扩展节点走到叶；
2. 叶是终端（已成五/满盘）→ 直接回传；
3. 否则加入 `pending`，打上 `waiting` 标记；
4. 攒够 `batch_size=32` 或结束 → `_flush`：一次前向评估所有叶子，展开并回传；
5. 每 16 次模拟检查一次 `deadline`。

`_flush` 对每个叶子：

- `encode` + `legal_mask`；
- `evaluate` 得到策略/价值；
- `_expand`：按 prior 取候选点前 `candidate_width=16` 建子节点；若某子节点落子即胜，标记 `terminal_value = -1.0`（对子节点而言是输）；
- `_backpropagate`：沿路径回传，每层取反。

### 5.4 展开

```text
moves = board.candidates(radius=1)
priors = policy[cell] for moves
归一化；若全为 0，用均匀分布
按 prior 降序取前 candidate_width
```

终端处理：

- 无候选点 → `terminal_value = 0`；
- 满盘 → `terminal_value = 0`；
- 子节点落子成五 → 子节点 `terminal_value = -1`。

### 5.5 根噪声

自对弈时 `add_noise=True`：

```text
noise ~ Dirichlet(alpha=0.3)
prior = (1 - 0.25) * prior + 0.25 * noise
```

只作用于根节点的子先验，用于探索。实际对局（`engine.py`）不加噪声。

### 5.6 结果

`_collect(root)`：

- `visits[cell] = child.visits`；
- `policy[cell] = child.visits / total`；
- `root_value = root.value_sum / root.visits`；
- `simulations = total`。

`best_move()` 取访问次数最多；`sampled_move(temperature)` 在温度 >0 时按 `visits^(1/T)` 采样。

## 6. 对局入口

`AlphaZeroEngine.choose_move`：

1. 空盘 → `(7,7)`；
2. 满盘 → `EngineError`；
3. `_forced_move`：在 radius 2 候选里找自己立即成五点，其次找对方立即成五点；有则直接下；
4. 否则 `searcher.search(...)`，`simulations` 默认 100,000，`time_budget` 默认 2.0s；
5. 温度默认 0 → `best_move()`；
6. 无着法 → `EngineError`。

`AZ_SIMULATIONS` 默认必须是 100,000，而不是 400。否则搜索在 400 次模拟后就停止，2s 时间预算从未生效；实测同一模型 400 次模拟 0 胜 8 负，放开后 6 胜 2 负。

## 7. 训练

### 7.1 预热（warmstart）

`teacher.collect_one_game`：

- 用 `classical_worker.search.Searcher` 作为教师；
- 每手取 `searcher.best_move`，再用 `ordered_moves(..., teacher_moves)` 得到排名；
- 策略目标：`exp(-rank/decay)`，位置 `cell = row*15+col`，`decay=2.0`；
- 合法掩码 radius 1；
- 价值目标：
  - `eval`（默认）：`tanh(position.evaluate(player) / 8000)`；
  - `outcome`：整局结束后 `±1`。

`warmstart.py`：

- `ProcessPoolExecutor` 并行收集（Windows spawn 需要 `teacher.py` 可导入）；
- 留 10% 样本做验证；
- 默认 `steps = max(1, len(examples)//32) * 6`；
- 每 50 步打印“验证首选一致率”；
- 保存权重和教师样本（`--data-out`）。

### 7.2 自对弈（selfplay）

`SelfPlay.play_game`：

- 黑先；
- 前 `temperature_moves=12` 手温度 1.0，之后 0；
- 每手 `search(add_noise=True)`，记录 `planes`、`policy`、`legal`；
- `board.play` 成五即结束；
- 结束后按胜方给每个样本 `value = ±1`，和局 0。

样本价值是“该样本的落子方最终是否获胜”，不是“当前局面相对父节点”。

### 7.3 训练循环

`train.py` 每轮：

1. 自对弈 `--games` 局，加入 `TensorDataset`；
2. 用一局自对弈做验证；
3. `train_gpu` 跑 `--train-steps` 步，批大小 1024（预热）/默认 256（自对弈入口）；
4. 评估验证策略/价值损失；
5. 保存检查点；
6. 可选保存回放缓冲区。

`--init` 与 `--model` 分开：从预热权重开始训练，但不覆盖预热文件。

`--teacher-data` 必须混入缓冲区。否则第一轮自对弈会把预热知识冲掉；实测“教师一致率”从 64% 提升到 76%。

### 7.4 GPU 数据管道

`TensorDataset` 把样本常驻 GPU：

| 张量 | dtype | 形状 |
| --- | --- | --- |
| `planes` | fp16 | `(capacity, 7, 225)` |
| `policy` | fp32 | `(capacity, 225)` |
| `legal` | bool | `(capacity, 225)` |
| `value` | fp32 | `(capacity,)` |

- `sample` 用 `torch.randint` 选索引；
- 8 种对称增强用 `gather` + `PERMUTATION_TABLE`，与旧的逐样本变换位精确一致；
- 缓冲区满时按时间顺序丢弃最旧样本（整块左移）。

`train_tensors` 的 bf16 autocast 在 bs=1024 时从 208ms 降到 85ms。

## 8. 擂台（arena）

`arena.py` 让 AZ 与传统引擎对弈：

- AZ 黑/白交替；
- 默认 AZ 100,000 模拟、2.0s；传统 d4/0.5s；
- 统计胜/负/和。

`warmstart --value-target outcome` 的模型 0 胜 8 负；改成 `eval` 后 6 胜 2 负。价值头方向错误会让 PUCT 退化到“只按先验选”，这是历史缺陷，当前默认已修正。

加入威胁平面后：

| 模型 | 平面 | 擂台 |
| --- | --- | --- |
| `warmstart_v2.pt` | 3 | 4 胜 4 负 |
| `warmstart_v3.pt` | 7 | 4 胜 4 负 |
| `current_v3.pt`（预热 + 50 轮自对弈） | 7 | 8 胜 0 负 |

预热阶段新增的 4 个通道从零开始，2000 步不足以让它们发挥作用；威胁平面的收益要在自对弈阶段才体现。8 局擂台分辨率有限，敏感指标是“验证首选一致率”和“强制堵点正确率”。

## 9. 参数

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `AZ_MODEL_PATH` | `models/az/current.pt` | 权重 |
| `AZ_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |
| `AZ_SIMULATIONS` | 100000 | 每手模拟上限 |
| `AZ_C_PUCT` | 1.5 | 探索常数 |
| `AZ_BATCH_SIZE` | 32 | 叶评估批大小 |
| `AZ_CANDIDATE_RADIUS` | 1 | 候选半径 |
| `AZ_CANDIDATE_WIDTH` | 16 | 根扩展保留的候选数 |
| `AZ_TIME_BUDGET` | 2.0 | 每手秒数 |
| `AZ_TEMPERATURE` | 0 | 0 为确定性选最大访问 |

每局 `config` 可覆盖 `simulations` / `temperature`。`model_path` 不存在时构造随机网络，便于跑通链路，但棋力无意义。

## 10. 运行顺序

```sh
# 1. 预热：传统引擎当老师，收集样本并训练
.venv/Scripts/python.exe -m az_worker.warmstart --games 300 --depth 4 --time 0.1 \
  --steps 2000 --teacher-moves 4 --decay 1.5 --workers 12 --batch-size 1024 \
  --model models/az/warmstart_v3.pt --data-out models/az/teacher_data_v3.pt

# 2. 自对弈：从预热权重开始，混入教师样本
.venv/Scripts/python.exe -m az_worker.train --init models/az/warmstart_v3.pt \
  --model models/az/current_v3.pt --teacher-data models/az/teacher_data_v3.pt \
  --games 10 --iterations 50 --lr 1e-4 --device cuda --batch-size 1024

# 3. 擂台
.venv/Scripts/python.exe -m az_worker.arena --model models/az/current_v3.pt \
  --games 8 --device cuda --candidate-width 16 --time-budget 2.0 \
  --classical-depth 4 --classical-time 0.5 --seed 42
```

依赖：

```sh
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install numpy
```

系统 Python 没有 torch，所有 `az_worker` 命令都用 `.venv/Scripts/python.exe`。

## 11. 已知边界

- 教师弱于对手会封顶策略上限；教师至少要与目标对手同强（d6/0.3s 对 d8/2.0s 仍偏弱）。
- 自对弈轮数是威胁平面收益的来源；纯预热不足以拉开差距。
- `candidate_width=16` 可能裁掉唯一防守点，虽然根节点强制手会先处理成五。
- value 头在极端局面（已成五、分数爆炸）仍可能饱和；`score_scale=8000` 是实测折中。
- 8 局擂台无法分辨小差异，评测应优先看一致率和堵点准确率，或扩大局数。
- 训练产物在 `models/`，被 `.gitignore` 排除；换机器需要重新训练或拷贝权重。
