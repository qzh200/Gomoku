# 引擎对比、评测与运维

这份文档回答三个问题：三个引擎各是什么性格、怎么衡量它们的棋力、线上出问题时先看哪里。

## 1. 三种引擎的定位

| 维度 | classical | llm | az |
| --- | --- | --- | --- |
| 决策核心 | 棋型评估 + PVS + VCF | 模型阅读棋形文字 | 策略/价值网络 + MCTS |
| 确定性 | 高（同参数同盘面基本同手） | 低（温度/服务端） | 高（温度 0） |
| 单步延迟 | 约 2s（可调） | 外部 API 延迟，通常 1–30s | 2s 预算或模拟数上限 |
| 成本 | CPU | 按 token 计费 | 本地 GPU/CPU |
| 离线 | 是 | 否 | 是（需权重） |
| 训练 | 无 | 无 | 自对弈训练 |
| 强项 | 战术、必堵/必胜、稳定 | 自然语言棋形推理、可解释 | 局面直觉、长期策略 |
| 弱项 | 叶评估粗糙、宽度裁剪 | 可能合法但瞎、费用/延迟 | 训练成本、权重依赖 |

## 2. 选型建议

- **要稳定、可复现、无外部依赖**：`classical`。默认配置已包含 VCF，能处理绝大多数“一步杀/必堵”局面。
- **要演示 LLM 下棋、或研究 Prompt**：`llm`。必须保留棋形文字和“这一手”行；只给坐标会退化为瞎下。
- **要研究学习型引擎、已有 GPU**：`az`。从随机网络自对弈训练；直接用未训练的权重没有意义。
- **AI 对战**：`classical` vs `az` 或 `classical` vs `llm` 最能体现差异；两个 `llm` 对局会产生双倍费用和长延迟。
- **契约/链路测试**：`mock_worker`，不参与正式对局。

## 3. 评测方法

### 3.1 传统引擎：benchmark

```sh
python -m classical_worker.benchmark --games 20 --budget 0.2
```

对冻结的 v1 引擎，交替先后手和 8 种开局。关注胜率、平均/p95 用时、完成深度。这是回归
“搜索改造是否变强”的主要工具。

### 3.2 AlphaZero：arena

```sh
.venv/Scripts/python.exe -m az_worker.arena --model models/az/current.pt \
  --games 40 --device cuda --time-budget 2.0 \
  --classical-depth 4 --classical-time 0.5 --opening-moves 2 --seed 42
```

对另一个权重也走同一个入口：

```sh
.venv/Scripts/python.exe -m az_worker.arena --model models/az/gen2.pt \
  --opponent az --opponent-model models/az/current.pt \
  --games 40 --device cuda --time-budget 2.0 \
  --opponent-simulations 100000 --opening-moves 2 --seed 42
```

局数不能太少，也不能关掉随机开局：温度 0 时同一对权重的对局完全确定，`--opening-moves 0`
会把 N 局压缩成 2 个棋局。8 局只够发现大缺陷；判断小差异要看固定局面的价值头符号，
或威胁平面上的堵点正确率（见 [AlphaZero 引擎](05-az-engine.md) §4）。

### 3.3 LLM：固定局面探针

对同一局面采样 3 次，检查落点是否落在预期集合：

| 局面 | 预期 |
| --- | --- |
| 己方有四 | 成五点 |
| 对方有四 | 堵点 |
| 对方斜活三 | 活三端点 |
| 空盘 | 中心附近 |
| 己方活三 | 延伸成活四 |

Prompt 改动必须重跑这组探针；只看一局完整对局会被随机性误导。

### 3.4 契约测试

`tests/contracts/` 会启动真实 Python Worker，对 `httptest` 服务完成：

- 领取 → 落子 → 版本推进；
- 租约续期/释放/过期；
- 幂等键复用；
- OpenAPI 契约字段；
- AI 对 AI 完整对局。

环境里没有 `python`、或者缺 `.venv` 与 torch 时，相应用例会跳过而不是失败。LLM 一侧用
桩服务替代真实模型，不需要 API Key。

## 4. 环境变量矩阵

### MyGo

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MYGO_ADDR` | `127.0.0.1:8080` | 监听地址 |
| `MYGO_CONFIG` | `configs/engines.json` | 静态引擎配置 |
| `MYGO_RUNTIME_CONFIG` | `configs/engines.runtime.json` | 运行时令牌文件 |

### 所有 Worker

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MYGO_URL` | `http://127.0.0.1:8080/api/v1` | API 根地址 |
| `MYGO_ENGINE_ID` | 引擎各自默认 | 必须与运行时配置中的 `engine_id` 一致 |
| `MYGO_ENGINE_TOKEN` | 空 | 为空时读运行时配置 |
| `MYGO_RUNTIME_CONFIG` | `configs/engines.runtime.json` | Worker 侧读取令牌的路径 |
| `MYGO_WORKER_ID` | `<engine_id>-worker-01` | 日志/租约标识 |
| `MYGO_WAIT_SECONDS` | 20 | 长轮询等待 |
| `MYGO_LEASE_EXTEND_SECONDS` | 120 | 续期长度 |

### classical

| 变量 | 默认 |
| --- | --- |
| `CLASSICAL_DEPTH` | 8 |
| `CLASSICAL_WIDTH` | 12 |
| `CLASSICAL_TIME_BUDGET` | 2.0 |
| `CLASSICAL_VCF_BUDGET` | 0.10 |
| `CLASSICAL_TT_ENTRIES` | 200000 |

### llm

| 变量 | 默认 |
| --- | --- |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | 必填 |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | 0.2 |
| `OPENAI_MAX_TOKENS` | 2000 |
| `OPENAI_TIMEOUT` | 60 |
| `OPENAI_MAX_ATTEMPTS` | 2 |
| `OPENAI_DISABLE_THINKING` | 0 |

### az

| 变量 | 默认 |
| --- | --- |
| `AZ_MODEL_PATH` | `models/az/current.pt` |
| `AZ_DEVICE` | auto |
| `AZ_SIMULATIONS` | 100000 |
| `AZ_BATCH_SIZE` | 32 |
| `AZ_TIME_BUDGET` | 2.0 |
| `AZ_TEMPERATURE` | 0 |
| `AZ_VCF_BUDGET` | 0.25 |
| `AZ_VCT_BUDGET` | 0.25 |
| `AZ_SOLVER_DEPTH` | 10 |
| `AZ_SOLVER_NODES` | 4000 |
| `AZ_CANDIDATE_RADIUS` | 0 |

每局 `config` 可以覆盖的键：

| 引擎 | 可覆盖 |
| --- | --- |
| classical | `depth`、`width`、`time_budget` |
| llm | 无 |
| az | `simulations`、`temperature` |

服务端不校验这些键，未知键会被静默忽略。

## 5. 运行手册

### 5.1 首次启动

```sh
GOPROXY=off go build ./...
go run ./cmd/app
```

控制台会输出：

```text
mygo: 未找到 configs/engines.json，使用内置引擎与随机令牌   # 仅当静态配置缺失
mygo: 引擎令牌已写入 configs/engines.runtime.json
mygo: API 已启动: http://127.0.0.1:8080/api/v1
mygo: 本地客户端令牌: ...
mygo: 引擎 az (...) 可用=true
mygo: 引擎 classical (...) 可用=true
mygo: 引擎 llm (...) 可用=true
```

引擎按 `engine_id` 排序输出。令牌只有在随机生成时才会打印（`token=...`），静态配置里
写死的令牌不会出现在日志里；两种情况下令牌都会写进 `configs/engines.runtime.json`。

随后另开终端启动 Worker。

### 5.2 启动 Worker

```sh
python -m classical_worker
python -m llm_worker
.venv/Scripts/python.exe -m az_worker
```

Worker 会在找到 `.env` 文件时打印 `参数文件: <path>`，随后打印 `开始轮询 <地址>`。
看到 `领取 ...` / `落子 ...` 说明链路正常。

### 5.3 创建对局

UI 直接操作，或调用 API：

```sh
curl -X POST http://127.0.0.1:8080/api/v1/games \
  -H 'X-Local-Client-Token: <local token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "ai_vs_ai",
    "players": {
      "black": {"type": "ai", "engine_id": "classical", "config": {"depth": 3}},
      "white": {"type": "ai", "engine_id": "az", "config": {"simulations": 4000}}
    }
  }'
```

响应是 `201`，正文为 `{"game": {...}, "owner_token": "..."}`。`owner_token` 只返回一次，
后续 `PATCH` / `DELETE` 用 `X-Game-Owner-Token` 带上它。

### 5.4 测试

```sh
GOPROXY=off go build ./...
GOPROXY=off go test ./... -count=1
python -m unittest discover -s classical_worker/tests -t .
python -m unittest discover -s llm_worker/tests -t .
.venv/Scripts/python.exe -m unittest discover -s az_worker/tests -t .
```

### 5.5 AlphaZero 训练

自对弈使用公共战术规则与可选 VCF/VCT，详见 [AlphaZero 引擎](05-az-engine.md)。
默认 25,000 步 / 1,000 轮，每轮至少 32 局、25 次梯度更新；开启八对称增强，
候选半径为 2，学习率在总进度 60%、85% 衰减。自对弈局数增加，耗时需重新测量。

建议使用新的权重路径和新的回放路径，从随机网络起训：

```powershell
.venv/Scripts/python.exe -m az_worker.train --device cuda --workers 8 --total-steps 25000 --iterations 1000 --games 32 --model models/az/tactical_v2.pt --buffer-file models/az/tactical_v2_replay.pt --seed 1
```

中断后使用相同命令加 `--resume`。如显存不足可降低 `--workers`，或改用
`--worker-device shared`。训练完成后，实战设置 `AZ_MODEL_PATH=models/az/tactical_v2.pt`，
可设 `AZ_CANDIDATE_RADIUS=2` 与训练一致。
CSV 写入权重同名 `.log`，TensorBoard 写入 `models/az/tensorboard/tactical_v2/`。

```powershell
.venv/Scripts/tensorboard.exe --logdir models/az/tensorboard --port 6006
```

棋力探针默认每 50 轮打 8 局，对手为传统引擎 d2/0.05s，候选 512 模拟、
VCF/VCT 各 0.1s；正式比较使用更多固定开局对战。
`--selfplay-vcf 0 --selfplay-vct 0` 关闭外部解算器，MCTS 内基础战术规则仍然保留。

## 6. 故障排查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| Worker 启动即报“拿不到 token” | MyGo 未启动或 engine_id 不匹配 | 先启动 MyGo；检查 `MYGO_ENGINE_ID` 与 `engines.runtime.json` |
| Worker 一直 `204` | 没有轮到它，或棋局暂停/结束 | 检查棋局状态和当前方 |
| 落子反复 `CLAIM_EXPIRED` | 计算超过租约 | 增大 `MYGO_LEASE_EXTEND_SECONDS`，或降低搜索预算；单次续期上限 300 秒、累计上限 900 秒，超出只能缩短计算 |
| 落子反复 `VERSION_CONFLICT` | 回合已被其他 Worker 完成/悔棋 | 丢弃该手，重新领取；确认同一引擎没有重复 Worker |
| 同一回合反复失败 | 引擎抛异常 → 释放 → 退避后重领 | 修复引擎；退避每次失败加 2 秒，60 秒封顶；LLM 解析失败应兜底而不是抛错 |
| LLM 总是下同一个错点 | 修正消息没有重复可行动作列表 | 检查 `INVALID_MOVE_HINT` 和重试消息 |
| LLM 回答为空 | 推理模型 token 预算耗尽 | 增大 `OPENAI_MAX_TOKENS`，或 `OPENAI_DISABLE_THINKING=1` |
| LLM 请求 400 | 服务端不认 `thinking` 字段 | 客户端会自动去掉重试；持续 400 检查模型名/URL |
| AZ 用 CPU 极慢 | 未装 CUDA 版 torch | 用 `.venv` 的 cu128 wheel，设置 `AZ_DEVICE=cuda` |
| AZ 每手固定不动 | `AZ_SIMULATIONS` 太小 | 保持 100000，让时间预算生效 |
| AZ 对传统引擎常年 0 胜 | 权重没训练到位，或对手档位不同 | 先看训练 CSV 的 `eval_winrate` 曲线；对比时固定 `--classical-depth/--classical-time` 与 `--eval-simulations` |
| 擂台每局完全相同、动辄 40:0 | 温度 0 下两个网络都是确定性的，且没加随机开局 | 保持 `--opening-moves 2`，并提高 `--games` |
| 棋盘文字/坐标错位 | 行列为 0 基，标签是 1 基 | 显示层用 `A`–`O`/`1`–`15`，引擎/API 用 0 基 |
| 端口占用 | 已有 MyGo 实例 | 关闭旧进程或设置 `MYGO_ADDR` |

## 7. 安全与数据

- API 默认只监听 `127.0.0.1`；不要改成 `0.0.0.0` 后暴露到公网。
- `configs/engines.runtime.json` 含明文引擎令牌与本地客户端令牌，权限 `0600`，已在 `.gitignore`。
- `.env` 含 `OPENAI_API_KEY`，已在 `.gitignore`。
- `models/` 被忽略；权重和训练数据不会进入 Git。
- `GET /games` 等读接口不需要凭据，但服务只监听回环地址；响应不包含令牌。
- 删除棋局不会清理磁盘（本来就没有持久化）。

## 8. 坐标与颜色速查

```text
内部/API/Worker:  (row, col), 0..14, 0 基
UI 显示:          A..O 列 + 1..15 行
board[row][col]:  0 空, 1 黑, 2 白
黑先手；五连即胜；无禁手
上一手:           last_move = {row, col, player, number}
版本:             每落一子/悔棋 +1
```

转换只发生在显示层：`ui.placeLabel(row, col)` 返回 `columnLabels[col] + rowLabels[row]`。引擎和 API 之间没有任何标签解析。

## 9. 代码地图

```text
cmd/app/                进程入口：配置、HTTP、窗口
internal/domain/gomoku/ 规则与棋局状态
internal/application/   用例、并发、幂等
internal/repository/    内存存储
internal/scheduler/     租约与回合队列
internal/transport/http REST API
internal/auth/          引擎注册表与令牌
internal/config/        静态/运行时配置
ui/                     原生桌面界面
assets/                 图标
aiworker/               Worker 协议与配置
classical_worker/       传统引擎
llm_worker/             LLM 引擎
az_worker/              AlphaZero 引擎
mock_worker/            测试用最小 Worker
tests/contracts/        Go 契约测试
api/openapi.yaml        API 契约
configs/                引擎配置示例/运行时文件
packaging/              发行包：启动器、PyInstaller 入口、构建/打包脚本
```

## 10. 新增引擎的端到端清单

1. 新建 `<engine>_worker/`，实现 `choose_move`。
2. 在 `config.py` 复用 `Connection.read(os.environ, "<engine_id>")`。
3. 在 `__main__.py` 用 `worker_dotenv_candidates` / `Worker`。
4. 在 `configs/engines.example.json` 增加条目。
5. 如静态配置缺失，`internal/config` 的内置列表也要增加（否则默认配置里没有它）。
6. 在 UI 引擎选择器中确认 `auth.Registry.Engines()` 能返回它；UI 自动读取，无需硬编码。
7. 增加 `.env.example`。
8. 增加单元测试与 Go 契约测试。
9. 在本文档与 `doc/README.md` 的引擎表中登记。

## 11. 发行包

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build.ps1
```

| 步骤 | 产物 |
| --- | --- |
| `go build ./cmd/app` | `dist/GomokuApp.exe` |
| `go build ./packaging/launcher` | `dist/Gomoku.exe` |
| `PyInstaller --onedir packaging/worker_entry.py` | `dist/GomokuWorkers/`（包内为 `runtime/GomokuWorkers.exe`） |
| `export_model.py` + `pack.ps1` | `dist/Gomoku/`、`dist/Gomoku-win64.zip` |

要点：

- 打包环境是 `.venv-pack` 里的 **CPU 版** torch；GPU 版 4.2 GB，发行包无法分发。
- 包内 `runtime/GomokuWorkers.exe` 与源码模式共用 `worker_entry.py --worker {classical,llm,az}`，所以用
  `--collect-submodules` 收集四个包，同时 `--exclude-module` 去掉训练侧代码。
- `dist/Gomoku/` 里的 `configs/engines.json` 由 `pack.ps1` 内联生成，令牌留空，主程序
  启动时随机生成并写 `engines.runtime.json`；启动器等这个文件出现后才拉起 Worker，
  最多等 15 秒，超时则结束主程序并退出。
- 发行包把 `packaging/README-发行版.md` 复制为 `使用说明.md`，并把 `az_worker/.env`
  的 `AZ_MODEL_PATH` 改写为 `models/az/run_last.pt`。
- `models/az/run_last.pt` 是裁剪过的推理权重（`extra` 对局棋谱已删除）。

