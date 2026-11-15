# MyGo 架构说明（V2.0 / API v1）

## 1. 总览

MyGo 是本地桌面五子棋平台，由一个 Go 进程承担三个角色：

| 角色 | 入口 | 与其他角色的交互 |
| --- | --- | --- |
| 桌面界面（Native UI） | `ui/` | 直接调用 `application` 服务（进程内） |
| RESTful API | `transport/http` | 暴露给本地客户端与 Python AI |
| Python AI Worker | 外部进程 | 只通过 RESTful API 访问棋局 |

```
        ┌──────────────────────────── MyGo 进程 ────────────────────────────┐
        │                                                                       │
        │  ui/ (MyGo Native UI + Lobe MyGo)                                       │
        │      │ 只做展示与交互                                                  │
        │      ▼                                                                │
        │  application/  ──用例编排──▶  domain/gomoku（规则）                   │
        │      │              ──读写──▶ repository/（棋局存储）                   │
        │      │              ──租约──▶ scheduler/（任务队列与租约）              │
        │      ▼                                                                │
        │  transport/http  ──协议转换──▶ application/                            │
        │      ▲                                                                 │
        │      │ 认证与授权                                                       │
        │  auth/（Owner Token / Engine Token）                                   │
        └──────┼───────────────────────────────────────────────────────────────┘
               │ 127.0.0.1:8080/api/v1
               ▼
     Python Classical AI / Python LLM AI / Mock Worker
```

关键边界：

* `domain/gomoku` 只含规则与领域状态，不 import 任何 http / repository / ui 包。
* `application` 编排用例，不出现 HTTP 状态码，也不出现 JSON 结构。
* `repository` 只抽象棋局读写；`scheduler` 只管理待领取回合与租约。
* `transport/http` 只做协议转换、输入校验、错误码映射。
* `ui` 只读取 `application` 返回的快照并把用户动作转成应用服务调用；
  UI **不得**直接修改棋盘或规则引擎状态。
* 游戏规则代码中不出现 `classical`、`llm` 等具体引擎字符串；引擎只出现在
  `auth` 注册表、`application` 的配置校验与 `ui` 的引擎下拉列表里。

## 2. 模块职责

### 2.1 `domain/gomoku`

* `board.go`：`Board`（15×15）、`Cell` 常量（0 空 / 1 黑 / 2 白）、坐标合法性。
* `rules.go`：四方向连续五子判定、平局判定。
* `game.go`：`Game` 实体与状态机（`playing`/`finished`、`running`/`paused`）。
* `move.go`：`Move`（row, col, player, number）。

规则细节（冻结）：

* 棋盘 15×15，坐标 0～14，黑棋先手。
* 横向、纵向、两条对角线连续至少 5 子获胜（超过 5 子也算赢）。
* 第一版无禁手。
* 棋盘满且无人获胜 → 平局。
* 终局后拒绝落子。

### 2.2 `application`

三个服务，对应三类用例：

| 服务 | 用例 |
| --- | --- |
| `GameService` | 创建、列表、查询、删除、暂停/恢复 |
| `MoveService` | 统一落子（人类与 AI 共用一条路径）、落子历史 |
| `ClaimService` | 领取回合、续期、释放；长轮询 |

`MoveService.Submit` 的校验顺序固定为：

1. 身份（Owner Token 或 Engine Token）与棋局归属；
2. 回合（当前行动方必须是提交者所属方）；
3. 落点（范围 → 是否被占用）；
4. 版本（`expected_version` 与当前 `version` 一致）；
5. 游戏状态（`status == playing`；`execution_state == paused` 直接拒绝）；
6. AI 租约（AI 提交时 `claim_id` 必须存在、有效、属于该棋局与该引擎、
   且版本一致）。

任一步失败都返回领域错误，由 `transport/http` 映射为契约中的错误码。

### 2.3 `repository`

```go
type GameRepository interface {
    Create(game *gomoku.Game) error
    Get(id string) (*gomoku.Game, bool)
    List() []*gomoku.Game
    Delete(id string) bool
    Update(id string, fn func(*gomoku.Game) error) error   // 原子更新
    Idempotency() IdempotencyStore
}
```

`memory.go` 是内存实现：单把互斥锁保护棋局表、幂等表与棋局内的 `owner_token` 哈希。
所有「读-改-写」都在 `Update` 内完成，避免竞态。

### 2.4 `scheduler`

* `queue.go`：待领取回合的队列。`scheduler.TurnSource` 是一个函数类型，
  `application` 用它把「哪些回合可以领取」的判断交给调度器，调度器只负责
  原子领取，不认识棋局。
* `lease.go`：租约注册表、原子领取、续期、释放、过期、频率限制。

回合的领取条件（棋局在进行中、当前行动方是 AI、引擎匹配、未暂停、无有效
租约）由 `application/coordinator.go` 的 `pendingTurns` 判断，它在协调器锁内
运行，因此领取是原子的。长轮询的广播通道同样在 `application/coordinator.go` 中
提供：长轮询等待时不持有任何锁。

租约模型：

```go
type Lease struct {
    ClaimID   string
    TurnID    string        // "<game_id>:<version>"
    GameID    string
    EngineID  string
    WorkerID  string
    Version   int
    Player    int
    ExpiresAt time.Time
}
```

原子领取：领取操作在「仓库锁 → 调度器锁」的固定顺序下完成，因此两个 Worker
同时轮询同一回合时，只有一个能创建租约。

### 2.5 `transport/http`

`net/http` + Go 1.22+ 的 `ServeMux` 模式路由，统一前缀 `/api/v1`。
错误响应统一为：

```json
{ "error": { "code": "VERSION_CONFLICT", "message": "棋局版本已经变化" } }
```

### 2.6 `auth`

* 引擎注册表：`engine_id` → 显示名、Bearer Token、可用状态、最近活跃时间。
* 棋局 Owner Token：创建棋局时生成，仅创建时返回一次，仓库内只存哈希。
* 服务端从 Bearer Token 反查 `engine_id`，**不信任** 请求参数中的引擎名。
* Owner Token 与 Engine Token 是两套独立凭据，互不通用。

## 3. 并发语义

| 操作 | 语义 |
| --- | --- |
| 领取回合 | 全流程持有协调器锁；同一回合同时最多一个有效租约 |
| 续期 | 校验租约归属与未过期后延长 `expires_at`；受单次上限与累计上限约束 |
| 释放 | 标记租约失效并立刻广播唤醒其他 Worker |
| 提交落子 | 在仓库锁内完成校验与落子；`version` 原子 +1 |
| 暂停 | 失效该棋局全部未完成租约，并阻止新的领取 |
| 长轮询 | **不持有**协调器锁等待；等待期间用广播通道唤醒，超时返回 204 |

长轮询实现：`ClaimService.Claim` 在一个循环里尝试领取，失败则等待
`scheduler.Notifier` 的广播通道或超时（`wait_seconds`），请求上下文取消时
立即返回。

## 4. 三个版本的语义

| 机制 | 解决的问题 | 失效表现 |
| --- | --- | --- |
| 租约（lease） | 同一个回合被重复领取 | `CLAIM_EXPIRED` / `CLAIM_INVALID` |
| 版本（version） | 过期落子写入棋盘 | `VERSION_CONFLICT` |
| 幂等键（Idempotency-Key） | 同一个请求被重复提交 | 返回首次结果 / `IDEMPOTENCY_KEY_REUSED` |

三者不可互相替代，必须同时存在。

## 5. 认证与部署边界

* HTTP 服务默认只监听 `127.0.0.1:8080`。
* 引擎凭据来自 `configs/engines.json`（示例见 `configs/engines.example.json`），
  该文件不提交；缺失时进程启动时随机生成并打印到控制台。
* 凭据不写入源码、不写入日志文件、不出现在任何棋局响应中。
* 当前安全边界建立在「仅本机监听 + 独立凭据」上；若要远程部署，必须在
  现有身份模型上增加正式的客户端认证与授权。

## 6. UI 架构

* 技术栈：`github.com/egoist/mygo`（Native UI，`ui` 包，GPU 绘制）+
  `github.com/lobehub/lobe-mygo`（组件库）。
* 棋盘由 MyGo Native UI 自定义绘制（`ui.Painter`），不是 Web 组件。
* 常规控件（按钮、选择器、标签、列表、提示）使用 Lobe MyGo 组件，保持一致视觉。
* 界面状态在 `ui` 包内维护，每帧从 `application` 拉取快照渲染。
* 界面动作一律转成应用服务调用（创建棋局、提交落子、暂停/恢复），UI 不直接改棋盘。

## 7. 测试策略

| 层次 | 测试 |
| --- | --- |
| 领域 | 四方向胜负、平局、非法落点 |
| 应用 | 三种模式生命周期、统一落子、错误回合、版本冲突 |
| 契约 | `tests/contracts` 用 `httptest` 打真实 HTTP 路由，校验 OpenAPI 字段 |
| 并发 | 两个 Worker 同时领取同一回合、租约过期重分配、暂停使租约失效 |
| 幂等 | 同键同体重试、同键异体冲突 |
| UI | `ui.NewTester` 无头渲染 + 点击，断言棋盘可交互 |
