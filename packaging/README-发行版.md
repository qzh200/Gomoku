# MyGo 五子棋 · 发行版

双击 `Gomoku.exe` 即可。它会依次启动：

1. `mygo.exe` —— 棋局服务 + 原生桌面界面（窗口标题「五子棋」）；
2. 三个 AI Worker —— `classical`（传统搜索）、`llm`（大模型）、`az`（AlphaZero）。

三个 Worker 在后台运行，**没有独立控制台窗口**；它们的输出见 `logs/`。
关闭窗口后，启动器和全部 Worker 一起退出。

## 目录

| 路径 | 内容 |
| --- | --- |
| `Gomoku.exe` | 启动器：先起 `mygo.exe`，等令牌写好后拉起 Worker，最后统一关停 |
| `mygo.exe` | 主程序（API + UI），可以单独运行，但不会启动 AI |
| `workers.exe` | 冻结的 Python 运行时，已含 torch/numpy，`--worker {classical,llm,az}` 选择引擎 |
| `configs/engines.json` | 引擎注册表（改为 `engines.example.json` 的内容：令牌留空，启动时随机生成） |
| `configs/engines.runtime.json` | 运行时令牌，每次启动重写；Worker 从这里读自己的 Bearer Token |
| `models/az/run_last.pt` | AlphaZero 权重（64 通道 × 6 残差块，训练 3500 步 / 4480 局） |
| `classical_worker/.env` | 传统引擎参数：`CLASSICAL_DEPTH`、`CLASSICAL_WIDTH`、`CLASSICAL_TIME_BUDGET` … |
| `llm_worker/.env` | 大模型参数：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL` … |
| `az_worker/.env` | AlphaZero 参数：`AZ_MODEL_PATH`、`AZ_DEVICE`、`AZ_TIME_BUDGET` … |
| `logs/` | 三个 Worker 的输出分别写到 `logs/{classical,llm,az}.log`（无控制台窗口） |

## 配置

**LLM 引擎必须先填 API Key**，否则该 Worker 启动后每手都失败：

```ini
# llm_worker/.env
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-flash
```

其余默认值即可开玩：

```ini
# az_worker/.env
AZ_MODEL_PATH=models/az/run_last.pt
AZ_DEVICE=auto            # 有 CUDA 用显卡，否则 CPU
AZ_TIME_BUDGET=2.0        # 每手思考秒数
```

传统引擎默认深度 8、宽度 12、每手 2 秒。

换成自己训练的权重：把 `.pt` 放进 `models/az/`，改 `AZ_MODEL_PATH` 即可。权重必须与当前网络结构一致（9 个输入平面、`channels`/`blocks` 记录在文件里）。

## 常见问题

- **窗口没出现**：`mygo.exe` 与 `Gomoku.exe` 必须在同一目录，且整个目录不要单独移动 exe。
- **某个引擎不动**：先看 `logs/<引擎>.log`；LLM 多半是 `.env` 里没填 Key，其余看端口 8080 是否被占用。
- **杀毒软件拦截**：PyInstaller 单文件包常被误报，加入白名单即可。
- **ALPHAZERO 很慢**：CPU 下每手按 `AZ_TIME_BUDGET` 走满，属正常。
