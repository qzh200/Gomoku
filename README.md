# MyGo 五子棋游戏平台

MyGo 是本地桌面五子棋平台。它由一个 Go 进程组成，承担三个角色：

* **桌面界面**（`egoist/mygo` Native UI + `lobehub/lobe-mygo` 组件），负责展示与交互；
* **权威棋局服务**（`application` + `domain/gomoku`），规则判定与落子只有这一条路径；
* **RESTful API**（`transport/http`），供 Python AI Worker 通过长轮询接入。

MyGo 是唯一的棋局状态管理者。Python AI 只做计算并提交落子，不维护棋局副本。

## 快速开始

```sh
go build ./...
go run ./cmd/app
```

启动后：

* API 监听 `http://127.0.0.1:8080/api/v1`（默认只监听回环地址）；
* 控制台打印**本地客户端令牌**与各引擎的注册信息；
* 桌面窗口打开，选择模式后点击「开始新棋局」。

地址可用环境变量覆盖：

```sh
MYGO_ADDR=127.0.0.1:9000 go run ./cmd/app
```

## 引擎配置

配置文件是 `configs/engines.json`，仓库里只提供 `configs/engines.example.json`：

```json
{
  "local_client_token": "随机本地令牌",
  "engines": [
    { "engine_id": "classical", "display_name": "Classical AI", "token": "随机令牌", "available": true },
    { "engine_id": "llm", "display_name": "LLM AI", "token": "随机令牌", "available": true }
  ]
}
```

* 文件缺失时使用内置引擎 `classical` 与 `llm`，所有令牌随机生成并打印在控制台。
* 令牌不会写入源码、公开日志或任何棋局响应。
* `engine_id` 是稳定标识；新增引擎只需改配置，游戏规则代码不依赖具体字符串。

## Python AI 接入

Worker 的完整协议见 [`docs/ai-protocol.md`](docs/ai-protocol.md)，接口契约见
[`api/openapi.yaml`](api/openapi.yaml)。最小循环：

```
POST /api/v1/ai/turn-claims   { "worker_id": "...", "wait_seconds": 20 }
  204 → 继续轮询
  201 → 解析 board，计算落点，必要时续期租约
POST /api/v1/games/{game_id}/moves
       { "row": 7, "col": 8, "expected_version": 12, "claim_id": "claim_..." }
```

仓库里的参考实现：

```sh
python mock_worker/mock_worker.py --engine classical --token <engine_token>
```

它只用标准库，完整演示了长轮询、租约使用、幂等提交与冲突处理。

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/games` | 创建棋局（返回一次性 `owner_token`） |
| GET | `/api/v1/games` | 查询棋局列表 |
| GET | `/api/v1/games/{id}` | 获取完整棋局 |
| DELETE | `/api/v1/games/{id}` | 删除棋局 |
| GET | `/api/v1/games/{id}/moves` | 查询落子历史 |
| POST | `/api/v1/games/{id}/moves` | 统一提交落子（人类与 AI 同一套规则） |
| PATCH | `/api/v1/games/{id}` | 暂停或恢复 AI 自动对战 |
| GET | `/api/v1/engines` | 获取可选择的 AI 引擎 |
| POST | `/api/v1/ai/turn-claims` | AI 长轮询并领取回合 |
| PATCH | `/api/v1/ai/turn-claims/{id}` | 续期任务租约 |
| DELETE | `/api/v1/ai/turn-claims/{id}` | 主动释放任务租约 |

## 架构

```
ui/ ──▶ application ──▶ domain/gomoku（规则）
              │  ├──▶ repository（棋局与幂等记录）
              │  └──▶ scheduler（租约与长轮询广播）
              ▼
transport/http ──▶ application
```

模块边界与并发语义见 [`docs/architecture.md`](docs/architecture.md)。

## 测试

```sh
gofmt -l .
go vet ./...
go test ./...
```

* `internal/domain/gomoku`：四方向胜负、平局、非法落点。
* `internal/scheduler`：租约原子领取、过期重分配、续期、释放、失效。
* `tests/contracts`：真实 HTTP 路由上的契约测试，覆盖三种模式、错误码、幂等、
  版本冲突、并发领取、暂停使租约失效，以及 Python mock worker 独立完成整局。
* `ui`：`ui.NewTester` 无头渲染与交互断言。

## 安全边界

* HTTP 默认只监听 `127.0.0.1`。
* 本地客户端令牌、棋局 Owner Token、引擎 Bearer Token 三套凭据互不通用。
* 当前安全模型建立在「仅本机监听 + 独立凭据」上；远程部署前必须增加正式的
  客户端认证与授权。
