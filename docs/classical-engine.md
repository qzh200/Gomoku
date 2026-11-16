# 传统算法引擎（negamax + alpha-beta + 贪心）

`classical_worker/` 是 `classical` 引擎的实现：不看模型的传统搜索，够快、可复现、
不花钱。它按 [`docs/ai-protocol.md`](ai-protocol.md) 的长轮询协议领任务，和 LLM
引擎共用同一套 Worker 协议。

## 1. 结构

```
aiworker/          两个引擎共用：领任务、续租约、提交落子、参数与令牌
classical_worker/
  position.py      棋盘、连子计分、落子/撤销、候选点
  search.py        negamax + alpha-beta + 迭代加深
  engine.py        把一次任务变成一手棋
  config.py        搜索参数（.env）
  __main__.py      命令行入口
```

`aiworker/` 里没有棋，`classical_worker/` 里没有网络：引擎只回答"下哪里"。

| 模块 | 职责 | 不做什么 |
| --- | --- | --- |
| `aiworker.protocol` | 领任务、租约、提交、退避 | 不懂棋 |
| `aiworker.settings` | .env、令牌、连接参数 | 不懂搜索 |
| `position` | 棋盘与评分，落子/撤销 | 不做搜索 |
| `search` | negamax、alpha-beta、候选排序 | 不碰网络 |
| `engine` | 组装一次搜索 | 不管协议 |

## 2. 算法

### 2.1 评分：连子的长度与开口

一局棋由"连子"组成，所以评分也按连子算。一段连子的价值取决于长度和**开口数**
（两端相邻的空点）；两头都被堵死的连子价值为 0。

| 长度 | 一个开口 | 两个开口 |
| --- | --- | --- |
| 1 | 1 | 2 |
| 2 | 50 | 200 |
| 3 | 500 | 5 000 |
| 4 | 10 000 | 100 000 |
| ≥5 | 胜（10 000 000） | 胜 |

`position.py` 只在落子/撤销时重算受影响的那四条线，总分随棋盘增量更新，所以
搜索里取分是 O(1)。

### 2.2 贪心：候选点与排序

只看"棋子附近的空点"，并按贪心价值排序：

```
价值 = 10 × 我这颗子能涨多少分 + 9 × 对手占这点能涨多少分
```

- 攻击与防守并重（防守权重略低），所以搜索先看威胁再看闲棋；
- 每层只保留前 `CLASSICAL_WIDTH` 个（默认 10），把分支因子压到可接受范围；
- 排序是 alpha-beta 生效的前提：好棋先看，后面的分支被剪掉。

排序之外还有两条直接的贪心规则，在搜索之前执行：

1. 我能连五 → 立刻连五；
2. 对手下一手能连五 → 必须堵（只有一个堵点时直接下）。

### 2.3 negamax + alpha-beta

negamax 让双方共用一份代码：每个节点都回答"轮到我走，这局面值多少"，回溯时
取负号。alpha-beta 记录"我已经能保证多少"与"对手不会让我更好"，一旦窗口关闭就
剪掉整枝。

```
value = max over moves of ( -negamax(child, depth-1, -beta, -alpha) )
```

细节：

- 越早的胜利分越高（`WIN - 剩余深度`），所以能赢就赢，不会拖着走；
- 时间预算用尽时抛 `SearchTimeout`，**落子与撤销都在 `finally` 里配对**，超时
  不会把棋盘弄脏；
- 迭代加深：2、4、6 层逐层搜，深一层没搜完就用上一层的结论，所以任何时刻都有
  一手可下的棋。

### 2.4 深度与时间

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `CLASSICAL_DEPTH` | 6 | 最大深度（层） |
| `CLASSICAL_WIDTH` | 10 | 每层保留的候选点数 |
| `CLASSICAL_TIME_BUDGET` | 2.0 | 单手思考时间上限（秒） |

中局 6 层约 0.3 秒；残局（30 子以上）多数能到 4 层，靠时间预算兜底。

## 3. 配置

```sh
cp classical_worker/.env.example classical_worker/.env
```

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `MYGO_URL` | 否 | `http://127.0.0.1:8080/api/v1` | MyGo API 根地址 |
| `MYGO_ENGINE_TOKEN` | 否 | 自动获取 | 留空则读 MyGo 写的 `configs/engines.runtime.json` |
| `MYGO_ENGINE_ID` | 否 | `classical` | 引擎标识 |
| `MYGO_WORKER_ID` | 否 | `classical-worker-01` | 日志标识 |
| `MYGO_WAIT_SECONDS` | 否 | `20` | 长轮询等待秒数 |
| `MYGO_LEASE_EXTEND_SECONDS` | 否 | `120` | 租约续期长度 |
| `CLASSICAL_DEPTH` | 否 | `6` | 最大搜索深度 |
| `CLASSICAL_WIDTH` | 否 | `10` | 每层候选点数 |
| `CLASSICAL_TIME_BUDGET` | 否 | `2.0` | 单手时间上限 |

开局时 MyGo 也可以按对局覆盖参数，例如创建棋局时给 `classical` 侧
`{"config": {"depth": 3}}`；对局参数优先于 .env。

## 4. 运行

```sh
go run ./cmd/app            # 先起 MyGo，它写令牌
python -m classical_worker
```

桌面界面里选「AI 对战」，先手或后手选 `Classical AI`。

## 5. 测试

```sh
python -m unittest discover -s classical_worker/tests -t .
go test ./tests/contracts/ -run TestClassicalWorkerFinishesAGame
```

- 单元测试覆盖评分（开口 vs 封闭）、落子/撤销复原、连五判定、堵四、候选排序，
  以及 **alpha-beta 的结果必须与不剪枝的 negamax 完全一致**（同时断言剪枝后的
  节点数确实更少）；
- 契约测试用两个真实进程（分别持 `classical` 与 `llm` 令牌）打完一整局真实 HTTP
  对局。

## 6. 棋力与限制

现状：默认参数下 9～12 手击败原来的贪心启发式，自己对自己 39 手分胜负，单手平均
0.27 秒。

已知不足：

- 评估是线性的，不识别双活三、四三这类组合杀，只能靠搜索深度发现；
- 没有置换表，重复局面的重新计算没有缓存；
- 没有必胜/必败的静态剪枝，靠 alpha-beta 与时间预算；
- 禁手规则未实现（与协议一致：第一版无禁手）。

第二代的设计见 [`classical-engine-v2.md`](classical-engine-v2.md)：窗口棋型表、
威胁强制着、Zobrist 置换表 + PVS、VCF/VCT，以及分阶段的棋力验收门槛。
