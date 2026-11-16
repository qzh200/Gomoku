# MyGo 五子棋游戏平台开发任务书 V2.0

**状态：建议作为首次正式开发基线**

**API 版本：v1**

**通信方式：RESTful API + Python 主动长轮询**

**核心原则：接口先行、高内聚低耦合、增量开发、阶段性 Git 提交**

## 1. 系统目标与职责

项目由三个独立程序组成。

| 组件                | 职责                                                   |
| ------------------- | ------------------------------------------------------ |
| MyGo                | 图形界面、棋局管理、规则判定、RESTful API、AI 任务分配 |
| Python Classical AI | 棋型评估、Minimax、Alpha-Beta 等传统算法               |
| Python LLM AI       | 调用 LLM 进行棋局分析和落子决策                        |

本任务仅开发 MyGo。

不在 Go 中实现 Python AI 算法，不要求 MyGo 启动、管理 Python 进程。

MyGo 不需要知道 Python 服务地址。Python AI 通过 HTTP 主动连接 MyGo。

### 1.1 系统权责

MyGo 是唯一的权威棋局状态管理者。

Python AI 不得直接修改棋盘，不得自行决定胜负，不得绕过 MyGo 提交棋局结果。

UI、API、AI Worker 需要共享同一套游戏规则和应用服务。

### 1.2 支持的对战模式

| 模式           | 标识             | 组成          |
| -------------- | ---------------- | ------------- |
| AI vs AI       | `ai_vs_ai`       | AI + AI       |
| Human vs AI    | `human_vs_ai`    | Human + AI    |
| Human vs Human | `human_vs_human` | Human + Human |

第一版 Human vs Human 仅要求同一客户端的本地双人对战。

三种模式必须复用相同的棋局实体、规则引擎和落子服务。

## 2. 固定游戏协议

### 2.1 棋盘规则

- 棋盘大小：15×15
- 空位置：0
- 黑棋：1
- 白棋：2
- 黑棋先手，双方交替落子
- `row`、`col` 从 0 开始，范围为 0～14
- 横向、纵向、对角线连续至少五子获胜
- 第一版采用无禁手规则
- 棋盘满且无人获胜时判定平局
- 游戏结束后拒绝继续落子

### 2.2 棋局状态

固定使用以下数据结构作为 JSON 契约：

```
{
  "id": "game_001",
  "mode": "ai_vs_ai",
  "players": {
    "black": {
      "type": "ai",
      "engine_id": "classical",
      "config": {"depth": 3}
    },
    "white": {
      "type": "ai",
      "engine_id": "llm",
      "config": {}
    }
  },
  "board": [],
  "current_player": 1,
  "status": "playing",
  "result": "ongoing",
  "winner": null,
  "last_move": null,
  "move_count": 0,
  "version": 0,
  "execution_state": "running",
  "ai_thinking": false
}
```

以上 `board` 为省略示例，实际所有棋局快照必须包含完整的 15×15 二维数组。

固定枚举：

- `status`：`playing`、`finished`
- `result`：`ongoing`、`black_win`、`white_win`、`draw`
- `execution_state`：`running`、`paused`
- `type`：`human`、`ai`

`winner` 在尚未获胜和平局时为 `null`，黑方获胜为 1，白方获胜为 2。

`version` 从 0 开始，每成功执行一次落子增加 1。暂停、恢复和 AI 任务领取不增加棋盘版本。

重新开局创建新 `game_id`，不复用旧棋局 ID。

`ai_thinking` 表示当前存在有效的 AI 决策任务租约，由 MyGo 计算并返回。

## 3. RESTful API 契约

统一前缀：

```
/api/v1
```

### 3.1 MyGo 最终接口清单

| 方法   | 路径                   | 说明                   |
| ------ | ---------------------- | ---------------------- |
| POST   | `/games`               | 创建棋局               |
| GET    | `/games`               | 查询棋局列表           |
| GET    | `/games/{id}`          | 获取完整棋局           |
| DELETE | `/games/{id}`          | 删除棋局               |
| GET    | `/games/{id}/moves`    | 查询落子历史           |
| POST   | `/games/{id}/moves`    | 统一提交落子           |
| PATCH  | `/games/{id}`          | 暂停或恢复 AI 自动对战 |
| GET    | `/engines`             | 获取可选择的 AI 引擎   |
| POST   | `/ai/turn-claims`      | AI 长轮询并领取回合    |
| PATCH  | `/ai/turn-claims/{id}` | 续期任务租约           |
| DELETE | `/ai/turn-claims/{id}` | 主动释放任务租约       |

所有这些路径均具有 `/api/v1` 前缀。

第一版不需要 WebSocket 或 SSE。

### 3.2 创建棋局

```
POST /api/v1/games
{
  "mode": "human_vs_ai",
  "players": {
    "black": {
      "type": "human"
    },
    "white": {
      "type": "ai",
      "engine_id": "classical",
      "config": {
        "depth": 3
      }
    }
  }
}
```

成功返回 HTTP 201：

```
{
  "game": {
    "id": "game_001",
    "mode": "human_vs_ai",
    "board": [],
    "current_player": 1,
    "status": "playing",
    "version": 0
  },
  "owner_token": "generated-game-owner-token"
}
```

这里 `game` 是完整棋局快照，上例省略其他字段。真实响应不得省略协议规定的字段。

`owner_token` 仅在创建时返回，用于授权本地界面操作该棋局，不能通过查询棋局接口再次公开。

### 3.3 获取棋局

```
GET /api/v1/games/{id}
```

返回第 2.2 节规定的完整棋局快照。

### 3.4 落子历史

```
GET /api/v1/games/{id}/moves
```

返回：

```
{
  "items": [
    {
      "row": 7,
      "col": 7,
      "player": 1,
      "number": 1
    },
    {
      "row": 7,
      "col": 8,
      "player": 2,
      "number": 2
    }
  ]
}
```

历史记录必须按实际落子顺序排列。

### 3.5 统一落子接口

```
POST /api/v1/games/{id}/moves
```

人类玩家请求：

```
{
  "row": 7,
  "col": 8,
  "expected_version": 12
}
```

AI 玩家请求：

```
{
  "row": 7,
  "col": 8,
  "expected_version": 12,
  "claim_id": "claim_abc123"
}
```

两者均需包含 HTTP 请求头：

```
Idempotency-Key: <唯一请求标识>
```

成功返回 HTTP 201：

```
{
  "move": {
    "row": 7,
    "col": 8,
    "player": 1,
    "number": 13
  },
  "game": {}
}
```

`game` 为落子后的完整棋局快照。

**该接口只有一套落子规则，不允许分别实现人类落子与 AI 落子的两套规则。**

服务端必须检查身份、回合、落点、版本、游戏状态和 AI 租约。

Idempotency-Key 按调用者和接口作用域去重。相同键、相同请求重试时返回已保存的结果，不得重复落子；相同键配不同请求内容返回 409。

## 4. AI 长轮询与任务租约

这是 V2.0 的核心变化。

### 4.1 引擎注册

MyGo 配置中维护 AI 引擎注册表。

每个 AI 具有：

- `engine_id`
- 显示名称
- 独立认证凭据
- 最近活跃时间
- 可用状态

第一版预置 `classical` 和 `llm`，但游戏规则代码不得依赖这两个具体字符串。

新增其他 AI 时，通过注册配置接入，无需修改核心游戏代码。

Python Worker 使用 Bearer Token 认证，服务端从认证信息确定其 `engine_id`，而不是信任请求参数中的引擎名称。

### 4.2 领取任务

```
POST /api/v1/ai/turn-claims
```

Python AI 主动发送长轮询请求：

```
{
  "worker_id": "classical-worker-01",
  "wait_seconds": 20
}
```

服务端检查是否存在符合该引擎身份的待处理 AI 回合。

有任务时，原子创建任务租约并返回 HTTP 201：

```
{
  "claim_id": "claim_abc123",
  "turn_id": "game_001:12",
  "expires_at": "2026-11-15T13:30:00Z",
  "game_id": "game_001",
  "version": 12,
  "player": 1,
  "board": [],
  "history": [],
  "last_move": null,
  "config": {
    "depth": 3
  }
}
```

实际 `board` 为完整棋盘，`history` 为完整历史。

`expires_at` 使用 UTC ISO 8601 时间格式。

如果 20 秒内没有任务，返回 HTTP 204，无响应体。

如果请求取消或客户端断开连接，应立即停止等待。

### 4.3 任务领取条件

仅当满足以下条件时才能分配任务：

1. 棋局正在进行。
2. 当前行动方是 AI。
3. 该 AI 的 `engine_id` 与 Worker 凭据匹配。
4. 对局没有暂停。
5. 当前回合不存在有效租约。
6. 棋局没有被删除。

同一个回合同时最多存在一个有效任务租约。

多个 Worker 同时轮询时，必须通过原子操作保证只有一个 Worker 成功领取该回合。

### 4.4 租约续期

```
PATCH /api/v1/ai/turn-claims/{id}
{
  "extend_seconds": 120
}
```

返回：

```
{
  "claim_id": "claim_abc123",
  "expires_at": "2026-11-15T13:32:00Z"
}
```

仅允许租约所属 Worker 所代表的授权 AI 引擎续期。

服务端应限制单次续期长度及累计租约时间。

LLM AI 可以在较长的模型推理期间定期续期。

### 4.5 主动释放租约

```
DELETE /api/v1/ai/turn-claims/{id}
```

当 AI 无法完成计算时，可以释放任务。

成功返回 HTTP 204。

释放后，如果棋局仍然符合分配条件，该回合可以重新进入待处理队列。

需要限制重复失败的任务领取频率，避免异常 Worker 无限占用 CPU 或产生大量 LLM 调用费用。

### 4.6 租约失效

以下情况需要使租约失效：

- 租约超时
- 棋局被删除
- 游戏结束
- 棋局暂停
- Worker 主动释放
- 该回合已经成功落子

租约失效后，迟到的 AI 结果必须被拒绝。

特别注意：

**租约避免重复领取，版本号避免过期落子，幂等键避免重复提交。三个机制解决的问题不同，不能相互替代。**

## 5. Python AI 运行流程

Python AI 采用统一 Worker 模式：

```
启动 Python AI
       |
       v
加载引擎配置和认证凭据
       |
       v
POST /ai/turn-claims
       |
       +-- 204 无任务
       |      |
       |      +--> 继续轮询
       |
       +-- 201 领取任务
              |
              v
          解析棋盘
              |
              v
          计算最佳落点
              |
              v
          必要时续期租约
              |
              v
       POST /games/{id}/moves
              |
              +-- 成功 --> 下一轮
              |
              +-- 冲突 --> 丢弃过期结果
              |
              +-- 失败 --> 根据错误类型重试或释放
```

Python AI 不需要自行维护棋局。

领取任务的响应已经包含完整棋盘及历史，所以正常情况下不需要再请求 `GET /games/{id}`。

MyGo 不负责 Python 算法耗时控制，但负责租约超时、过期结果拒绝以及服务端资源保护。

## 6. 三种模式的行为

### 6.1 Human vs Human

UI 根据当前回合提交落子。

不产生 AI 任务。

### 6.2 Human vs AI

人类落子成功后，如果下一方是 AI，MyGo 将该轮次标记为可领取。

对应的 Python Worker 通过长轮询获取任务、计算并提交。

如果 AI 执黑，棋局创建后第一回合立即可领取。

### 6.3 AI vs AI

创建棋局后，如果执行状态为 `running`，当前回合立即可领取。

每次 AI 落子成功后，下一 AI 回合自动变为可领取。

MyGo 不需要通过后台循环主动请求 Python。

### 6.4 暂停和恢复

```
PATCH /api/v1/games/{id}
{
  "execution_state": "paused",
  "expected_version": 12
}
```

仅 AI vs AI 模式支持手动暂停与恢复。

暂停必须阻止新的任务领取，并使当前尚未完成的 AI 租约失效。

恢复时重新开放当前回合的任务领取，不修改棋盘版本。

## 7. 认证与接口安全

第一版定位为本地桌面游戏平台。

HTTP 服务默认只监听 `127.0.0.1`，不得默认暴露到公网。

需要区分以下身份：

| 身份          | 凭据              | 权限                   |
| ------------- | ----------------- | ---------------------- |
| MyGo 本地界面 | 棋局 Owner Token  | 操作自己创建的棋局     |
| Classical AI  | 独立 Engine Token | 领取和提交所属 AI 回合 |
| LLM AI        | 独立 Engine Token | 领取和提交所属 AI 回合 |

本地界面通过内部应用服务创建棋局，或通过受本地客户端保护的创建入口调用 REST API。

Owner Token 不得用于冒充 AI，Engine Token 不得用于冒充人类玩家。

认证凭据不得写入源代码、公开日志或棋局响应。

如果以后支持远程部署，应在现有身份模型上增加正式的客户端认证与授权，不能依赖仅本机监听提供的安全边界。

## 8. 错误处理协议

统一错误格式：

```
{
  "error": {
    "code": "VERSION_CONFLICT",
    "message": "棋局版本已经变化"
  }
}
```

固定主要错误码：

| HTTP 状态 | 错误码                               |
| --------- | ------------------------------------ |
| 400       | `INVALID_REQUEST`                    |
| 401       | `UNAUTHORIZED`                       |
| 403       | `FORBIDDEN` / `NOT_YOUR_TURN`        |
| 404       | `GAME_NOT_FOUND` / `CLAIM_NOT_FOUND` |
| 409       | `VERSION_CONFLICT`                   |
| 409       | `POSITION_OCCUPIED`                  |
| 409       | `GAME_FINISHED`                      |
| 409       | `GAME_PAUSED`                        |
| 409       | `CLAIM_EXPIRED`                      |
| 409       | `CLAIM_INVALID`                      |
| 422       | `INVALID_GAME_CONFIG`                |
| 422       | `POSITION_OUT_OF_RANGE`              |
| 500       | `INTERNAL_ERROR`                     |

长轮询超时正常返回 204，不视为服务错误。

过期任务不得被自动转换成随机落子。

AI 返回非法坐标时不修改棋局，可将错误返回给 Worker，由 Worker 决定重新计算或释放任务。

## 9. MyGo 模块架构

建议目录结构：

```
mygo/
├── cmd/
│   └── app/
│       └── main.go
├── internal/
│   ├── domain/
│   │   └── gomoku/
│   │       ├── board.go
│   │       ├── rules.go
│   │       ├── game.go
│   │       └── move.go
│   ├── application/
│   │   ├── game_service.go
│   │   ├── move_service.go
│   │   └── claim_service.go
│   ├── repository/
│   │   ├── interfaces.go
│   │   └── memory.go
│   ├── scheduler/
│   │   ├── queue.go
│   │   └── lease.go
│   ├── transport/
│   │   └── http/
│   │       ├── router.go
│   │       ├── game_handler.go
│   │       ├── claim_handler.go
│   │       └── errors.go
│   └── auth/
│       └── token.go
├── ui/
├── api/
│   └── openapi.yaml
├── docs/
│   ├── architecture.md
│   ├── ai-protocol.md
│   └── decisions/
├── tests/
│   └── contracts/
├── configs/
├── go.mod
└── README.md
```

### 强制模块边界

- `domain/gomoku` 只能处理游戏规则和领域状态。
- `application` 编排业务用例。
- `repository` 抽象棋局读写。
- `scheduler` 管理待领取任务与租约。
- `transport/http` 只负责协议转换和输入校验。
- `auth` 处理身份和授权。
- `ui` 只负责展示与用户交互。

不允许规则引擎直接访问 HTTP、具体存储或 UI。

不允许 UI 绕过应用服务直接修改棋盘。

游戏规则引擎不应该包含任何 Classical AI 或 LLM AI 的专用逻辑。

### 并发要求

棋局更新与租约领取必须保证原子性。

HTTP 长轮询不得长时间持有棋局写锁。

任务领取、续期、提交落子和暂停等操作必须具有明确的并发语义。

## 10. 接口冻结与测试要求

在开始主要业务实现之前必须完成：

1. `api/openapi.yaml`
2. `docs/architecture.md`
3. `docs/ai-protocol.md`
4. JSON 请求与响应样例
5. 统一错误码
6. 基础契约测试

V1 API 冻结以后：

- 不允许任意修改接口路径。
- 不允许任意修改已有字段名称。
- 不允许修改字段语义和坐标规范。
- 不允许让 Classical AI 和 LLM AI 使用不同的游戏协议。
- 兼容性新增字段应保持可选。
- 破坏性变更必须经明确批准，并升级 API 版本。

至少测试：

- 三种对战模式创建及运行
- 四个方向胜负判断
- 平局、非法落子及错误回合
- 多棋局状态隔离
- 两个 Worker 同时领取同一任务
- 租约过期重新分配
- 租约续期与释放
- AI 计算期间暂停棋局
- 幂等请求重复提交
- 版本冲突
- Worker 异常退出
- AI vs AI 自动完成整局
- LLM 长时间思考期间的租约续期

所有 OpenAPI 契约应被自动化测试覆盖，保证 Python Agent 可以独立依照文档开发。

## 11. Git 开发要求

必须分阶段开发和提交。

推荐提交顺序：

```
docs: finalize v1 REST and AI polling contracts
chore: initialize Go project architecture
feat(domain): implement gomoku rules and board state
feat(application): implement game lifecycle and moves
feat(api): implement RESTful game resources
feat(scheduler): implement atomic AI turn claiming
feat(scheduler): implement lease renewal and expiry
feat(ui): implement three game modes and board UI
test: add concurrency and contract tests
docs: complete deployment and integration guide
```

要求：

- 每完成一个独立功能并通过相关测试后提交。
- 提交前执行 `gofmt`、`go test ./...`。
- 适当执行 `go vet ./...` 和并发测试。
- 不允许所有功能一次性提交。
- 不允许提交密钥或编译产物。
- 不允许覆盖用户已有的无关代码。
- 不擅自 `git push` 或强制重写历史。
- 提交消息必须描述实际改动。

## 12. 最终交付

Agent 应交付：

- 完整可运行的 MyGo 游戏程序
- 三种对战模式 UI
- 正确的五子棋规则引擎
- RESTful API
- AI 长轮询任务机制
- 任务租约及棋局版本校验
- 引擎注册和认证
- OpenAPI 文档
- Python AI 协议文档
- Mock Python Worker
- 自动化测试
- Git 开发记录
- README 和启动说明

正式 Python Classical AI 和 Python LLM AI 不属于本任务范围。

**最终验收重点：只需要依据协议编写 Python Worker，即可独立接入 MyGo，无须修改 MyGo 游戏规则或 HTTP 接口。**





在现有《MyGo 五子棋游戏平台开发任务书 V2.0》的基础上，补充以下技术要求：

**UI 技术栈正式确定为** `**egoist/mygo**` **+** `**lobehub/lobe-mygo**`**，使用纯 Go Native UI。**

开发前必须阅读并参考：

- https://github.com/egoist/mygo
- https://github.com/egoist/mygo/blob/main/AGENTS.md
- https://awesomemygo.dev/
- https://mygo.lobehub.com/
- https://mygo.lobehub.com/skills.md
- https://mygo.lobehub.com/llms.txt
- https://mygo.run/docs/ui/drawing

### 开发要求

1. 使用 Lobe MyGo 实现常规界面组件，保持一致的视觉设计。
2. 使用 MyGo Native UI 自定义绘制 15×15 棋盘。
3. 将棋盘绘制、用户交互、规则引擎、游戏服务及 RESTful API 严格解耦。
4. 桌面 UI 通过应用服务访问棋局，Python AI 通过 RESTful API 访问棋局，二者共享同一套业务规则。
5. 不允许在 UI 组件中直接修改棋局的权威状态。
6. 不允许引入不必要的 Web 前端工程或 JavaScript 运行时。
7. 开发前验证 Go 版本、MyGo 与 Lobe MyGo 的兼容性，并固定依赖版本。
8. 实现过程中必须查阅组件官方文档，禁止使用未经确认的组件 API。
9. 保持既定 RESTful API 与 AI 长轮询协议不变。
10. 严格遵循模块解耦、自动化测试和阶段性 Git 提交要求。

优先完成能够编译运行的最小原生窗口、棋盘绘制和交互验证，再继续开发完整业务功能。

**所有外部接口必须以冻结的 OpenAPI 契约为准，不允许因为 UI 实现需要而随意调整。**