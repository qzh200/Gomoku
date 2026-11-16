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
| `MYGO_ENGINE_TOKEN` | **是** | — | `llm` 引擎的 Bearer Token，取自 `configs/engines.json` |
| `MYGO_WORKER_ID` | 否 | `llm-worker-01` | Worker 标识，只用于日志 |
| `MYGO_WAIT_SECONDS` | 否 | `20` | 长轮询等待秒数（0–120） |
| `MYGO_LEASE_EXTEND_SECONDS` | 否 | `120` | 租约续期长度 |
| `OPENAI_BASE_URL` | 否 | `https://api.openai.com/v1` | 模型服务根地址 |
| `OPENAI_API_KEY` | **是** | — | 模型服务的 API Key |
| `OPENAI_MODEL` | 否 | `gpt-4o-mini` | 模型名 |
| `OPENAI_TEMPERATURE` | 否 | `0.2` | 采样温度 |
| `OPENAI_MAX_TOKENS` | 否 | `200` | 单次回答的 token 上限 |
| `OPENAI_TIMEOUT` | 否 | `60` | 请求超时秒数 |
| `OPENAI_MAX_ATTEMPTS` | 否 | `2` | 答案不合法时的重试次数 |

`MYGO_ENGINE_TOKEN` 与 `configs/engines.json` 里 `engine_id` 为 `llm` 的
`token` 必须是同一个值，否则服务端返回 401。

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
# 1. 启动 MyGo，它会打印本地令牌和引擎令牌
go run ./cmd/app

# 2. 填好 llm_worker/.env
# 3. 启动 LLM Worker
python -m llm_worker
```

桌面界面里选「AI 对战」，把先手或后手指向 `LLM AI`，开始新棋局即可。

## 4. 工作方式

1. Worker 用 `POST /api/v1/ai/turn-claims` 长轮询领任务；没有任务时服务端
   返回 204，Worker 立即继续下一轮。
2. 领到任务后，Worker 把棋盘画成文本（列 A–O、行 1–15，与界面上的标签
   一样），连同落子历史和当前执子方一起发给模型。
3. 模型只回答一段 JSON：`{"row": 7, "col": 8}`（0 基坐标，与 API 一致）。
4. Worker 校验落点：越界或已有棋子时，把原因回给模型让它重选，最多
   `OPENAI_MAX_ATTEMPTS` 次。
5. 模型思考期间，后台线程按 `MYGO_LEASE_EXTEND_SECONDS` 续期租约，慢模型也
   不会因为超时丢掉这一手。
6. 落子通过 `POST /api/v1/games/{id}/moves` 提交，必带 `Idempotency-Key`。
   409 表示棋局已经变了，结果丢弃、继续轮询，不会改成随机落子。

模型算不出来时，Worker 主动释放租约（`DELETE /api/v1/ai/turn-claims/{id}`），
这一手重新回到待领取队列，别的 Worker 还能接手。

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
* Worker 不判断胜负，也不改棋盘，所有规则由 MyGo 判定。
* 回答不合法时只重试固定次数，不做无限循环。
