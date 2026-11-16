# LLM 引擎接入说明

LLM 引擎是一个 Python Worker：它按 [`docs/ai-protocol.md`](ai-protocol.md) 的
长轮询协议向 MyGo 领任务，用 **OpenAI 兼容协议**向模型服务提问，把答案变成
一个合法落点再提交回去。

MyGo 不启动、不管理这个进程；两者只通过 RESTful API 通信。

## 1. 结构

```
llm_worker/
  __init__.py        包说明
  __main__.py        命令行入口
  config.py          参数：读 .env 与环境变量
  board.py           棋盘绘制与坐标解析
  openai_client.py   OpenAI 兼容协议的客户端
  engine.py          向模型要落点，并检查答案
  worker.py          领任务、续租约、提交落子
  .env.example       参数模板
  tests/             单元测试
```

各层互不依赖实现细节：

| 模块 | 职责 | 不做什么 |
| --- | --- | --- |
| `config` | 读取参数 | 不认识 MyGo，也不认识模型 |
| `board` | 画棋盘、解析坐标 | 不发请求 |
| `openai_client` | 说 OpenAI 协议 | 不懂五子棋 |
| `engine` | 组织提示词、校验答案 | 不管租约和轮询 |
| `worker` | 领任务、续租约、提交 | 不碰模型细节 |

换模型服务只改 `OPENAI_BASE_URL`；换玩法策略只改 `engine.py`；换协议版本只改
`worker.py`。

## 2. 配置

参数写在 `.env` 文件里，模板见 `llm_worker/.env.example`：

```sh
cp llm_worker/.env.example llm_worker/.env
```

`.env` 不提交（见 `.gitignore`）。Worker 按 `llm_worker/.env` → 仓库根 `.env` →
当前目录 `.env` 的顺序查找第一个存在的文件；也可以显式指定：

```sh
python -m llm_worker --env-file llm_worker/.env
```

环境变量优先于 `.env`，所以临时改一个参数不用动文件：

```sh
OPENAI_MODEL=deepseek-chat python -m llm_worker
```

### 参数一览

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `MYGO_URL` | 否 | `http://127.0.0.1:8080/api/v1` | MyGo API 根地址 |
| `MYGO_ENGINE_TOKEN` | 否 | 自动获取 | 引擎的 Bearer Token；留空则从运行时配置文件读 |
| `MYGO_ENGINE_ID` | 否 | `llm` | Worker 使用的引擎标识，对应 `engine_id` |
| `MYGO_RUNTIME_CONFIG` | 否 | `configs/engines.runtime.json` | MyGo 写运行时令牌的文件 |
| `MYGO_WORKER_ID` | 否 | `llm-worker-01` | Worker 标识，只用于日志 |
| `MYGO_WAIT_SECONDS` | 否 | `20` | 长轮询等待秒数（0–120） |
| `MYGO_LEASE_EXTEND_SECONDS` | 否 | `120` | 租约续期长度 |
| `OPENAI_BASE_URL` | 否 | `https://api.openai.com/v1` | 模型服务根地址 |
| `OPENAI_API_KEY` | **是** | — | 模型服务的 API Key |
| `OPENAI_MODEL` | 否 | `gpt-4o-mini` | 模型名 |
| `OPENAI_TEMPERATURE` | 否 | `0.2` | 采样温度 |
| `OPENAI_MAX_TOKENS` | 否 | `2000` | 单次回答的 token 上限 |
| `OPENAI_TIMEOUT` | 否 | `60` | 请求超时秒数 |
| `OPENAI_MAX_ATTEMPTS` | 否 | `2` | 答案不合法时的重试次数 |
| `OPENAI_DISABLE_THINKING` | 否 | `0` | 填 `1` 关闭模型的思考模式 |

### 令牌自动获取

MyGo 启动时把生效的引擎注册表写到 `configs/engines.runtime.json`（权限 0600，
不进版本库），Worker 直接从那里读自己的令牌，**不用手工复制**：

```
MyGo 启动 → 写 configs/engines.runtime.json
Worker 启动 → 读该文件，取 engine_id 对应的 token
```

查找顺序：`MYGO_ENGINE_TOKEN`（显式设置）→ `MYGO_RUNTIME_CONFIG` 指定的文件 →
`configs/engines.runtime.json`。用别的 MyGo 实例时，把 `MYGO_URL` 和
`MYGO_ENGINE_TOKEN` 一起填上即可。

MyGo 重启会换掉随机生成的令牌，所以**先启动 MyGo，再启动 Worker**。

### 推理模型（deepseek-flash、deepseek-reasoner 等）

推理模型会把 token 预算花在思考上，回答正文可能为空，只有 `reasoning_content`
里有内容。两种应对方式：

* `OPENAI_DISABLE_THINKING=1` —— 请求里带 `thinking: {"type": "disabled"}`，
  关闭思考。落子这种短任务本来就该直接回答，最省也最快。服务不认这个参数时
  Worker 会自动去掉它重试一次。
* 调大 `OPENAI_MAX_TOKENS` —— 思考也要预算，默认 2000。

即使如此，Worker 也做了兜底：`content` 为空时读 `reasoning_content`；思考被
截断（`finish_reason` 是 `length`）时报出明确原因，而不是笼统的「回答是空的」。

### 兼容的服务

只依赖 OpenAI 的 `POST {base}/chat/completions`，所以下面这些都能直接用：

| 服务 | `OPENAI_BASE_URL` |
| --- | --- |
| OpenAI | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Moonshot | `https://api.moonshot.cn/v1` |
| Ollama | `http://127.0.0.1:11434/v1` |
| LM Studio | `http://127.0.0.1:1234/v1` |
| vLLM | `http://127.0.0.1:8000/v1` |

## 3. 运行

```sh
# 1. 启动 MyGo，它会写好 configs/engines.runtime.json
go run ./cmd/app

# 2. 填好 llm_worker/.env（至少要有 OPENAI_API_KEY）
# 3. 启动 LLM Worker
python -m llm_worker
```

Worker 会自己从 `configs/engines.runtime.json` 取令牌，所以第 1 步必须先做。

桌面界面里选「AI 对战」，把先手或后手指向 `LLM AI」，开始新棋局即可。

## 4. 工作方式

1. Worker 用 `POST /api/v1/ai/turn-claims` 长轮询领任务；没有任务时服务端
   返回 204，Worker 立即继续下一轮。
2. 领到任务后，Worker 把棋盘画成文本（列 A–O、行 1–15，与界面上的标签
   一样），连同落子历史和当前执子方一起发给模型。
3. **坐标一律用棋盘上的标注**：模型回答 `{"move": "M5"}`，`M5` 就是棋盘上
   第 13 列第 5 行的交点。**不要给模型 0 基下标**——模型对着 1 基的棋盘回答，
   0 基的解析会让棋子整体错开一行一列，看起来就像"AI 乱下"。
4. **告诉模型哪些点能下**：空点会按"离已有棋子近"排序后列出来，模型只能从
   列表里选。这一步消除了"选了已有棋子的点"这种最常见的错误。
5. **后端替模型数连子**：模型很难在一堆 `.` 里数出三连四连，所以后端扫描
   对手的连子，把"对手在 M6/M7/M8 有三连，能挡的点：M5、M9"直接写进提示词。
   落子的选择仍然归模型，后端只负责数格子。
6. 模型思考期间，后台线程按 `MYGO_LEASE_EXTEND_SECONDS` 续期租约，慢模型也
   不会因为超时丢掉这一手。
7. 落子通过 `POST /api/v1/games/{id}/moves` 提交，必带 `Idempotency-Key`。
   409 表示棋局已经变了，结果丢弃、继续轮询，不会改成随机落子。

回答不合法（不是 JSON、点已有棋子、不在可下点列表里）时，Worker 把**具体原因
连同候选点列表**回给模型让它重选，最多 `OPENAI_MAX_ATTEMPTS` 次——小模型很容易
在长提示词里弄丢列表，把它贴在纠正消息里比重述规则有用得多。

`OPENAI_MAX_ATTEMPTS` 次都还不合规时，引擎**改用候选列表的第一位**落子并打日志
（`[llm] 模型 N 次都没给出可下点，改用 X`）。这一手可能不如模型选的，但比丢掉
这一手好：丢手会让服务端把同一个回合再派下来，形成"重试—失败—退避"的循环。

失败不会让 Worker 退出，但会**逐步退避**（2 秒起步，翻倍到 60 秒封顶），这样
服务坏了也不会被反复追问。

### 为什么不是"模型不行"

同一局面上做过对照：只问 `{"row": 0-14, "col": 0-14}` 时，模型给出的坐标
按 0 基解析会落到**已有棋子**的格子上（例如白棋自己的 L7），Worker 只能反复
重问；换成棋盘坐标 `M5` 再补上"黑棋 M6/M7/M8 三连，能挡的点 M5/M9"，同一个
模型连续给出正确防守。

所以"AI 乱下"的第一责任人通常是**提示词与解析器的坐标系不一致**，而不是模型
本身。模型仍然会下出弱手（例如没在早期堵住对手的三连），这是模型能力问题，
和协议、坐标无关。

### 模型不配合时

小模型会反复给出同一个不可下的点（例如战场在 H 列，它却一直说 `B6`）。这套
处理让它不再拖垮对局：

| 现象 | 处理 |
| --- | --- |
| 答案里没有可解析的坐标 | 回 `请只输出 JSON`，并重贴候选点 |
| 点是已有棋子 | 回 `D4 该位置无效或已有棋子`，并重贴候选点 |
| 点在候选列表外 | 回 `B6 不在可下点里`，并重贴候选点 |
| 仍然不配合 | 用候选列表第一位落子，日志写明"模型被越过" |

实测（同一局面 4 次）：模型 4/4 从候选里选；一整局 LLM 自战与 LLM 对传统算法
各 1 局，`本轮失败` 与回退行数均为 0，不再出现"重试—退避"的长循环。

## 5. 测试

单元测试只用标准库：

```sh
python -m unittest discover -s llm_worker/tests -t .
```

契约测试让真正的 Python Worker 对着真正的 MyGo API 下棋，模型服务由
`tests/contracts/llm_worker_test.go` 里的桩实现：

```sh
go test ./tests/contracts/ -run TestLLMWorkerFinishesAGame
```

## 6. 成本与限制

* 每手棋一次模型调用，费用与模型单价一致；长对局要留意 token 消耗。
* 提示词带整张棋盘和完整历史，对局越长请求越大。
* 推理模型不关思考时，一次落子可能消耗几千 token；`OPENAI_DISABLE_THINKING=1`
  能省下大部分。
* Worker 不判断胜负，也不改棋盘，所有规则由 MyGo 判定。
* 回答不合法时只重试固定次数，不做无限循环。
* 令牌在 `configs/engines.runtime.json` 里，是明文；该文件不进版本库，但要
  按凭据对待。
