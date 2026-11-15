# Python AI 接入协议（API v1）

本文是 Python Worker 的完整接入说明。Worker 只需遵守本文即可独立开发，
**不需要修改 MyGo 的游戏规则或 HTTP 接口**。

## 0. 角色边界

* MyGo 是唯一权威棋局状态管理者。
* Python AI 不维护棋局副本的权威性，只做计算并提交落子。
* Python AI 不决定胜负，不直接修改棋盘。

## 1. 认证

每个引擎（AI）持有独立的 Bearer Token：

```
Authorization: Bearer <engine_token>
```

服务端从 Token 解析出 `engine_id`，**忽略**请求参数里任何自称的引擎名。
Token 由 MyGo 配置文件 `configs/engines.json` 提供：

```json
{
  "engines": [
    {
      "engine_id": "classical",
      "display_name": "Classical AI",
      "token": "replace-me",
      "available": true
    },
    {
      "engine_id": "llm",
      "display_name": "LLM AI",
      "token": "replace-me-too",
      "available": true
    }
  ]
}
```

`engine_id` 是稳定标识。游戏规则代码不依赖具体字符串；新增引擎只需在此注册。

## 2. Worker 主循环

```
启动
  ↓
加载引擎配置与凭据
  ↓
POST /api/v1/ai/turn-claims   { "worker_id": "...", "wait_seconds": 20 }
  ↓
  ├─ 204 No Content（没有任务）
  │      → 立即继续下一次轮询（建议间隔 0～1 秒，避免空转）
  │
  └─ 201 Created（领取到任务）
         ↓
     解析 board / history / config
         ↓
     计算最佳落点（期间按需续期租约）
         ↓
     POST /api/v1/games/{game_id}/moves
         ↓
     ├─ 201 → 继续下一轮轮询
     ├─ 409 VERSION_CONFLICT / CLAIM_EXPIRED → 丢弃过期结果，继续轮询
     └─ 其他错误 → 按错误类型决定重试或释放租约
```

`worker_id` 用于标识 Worker 实例（例如 `classical-worker-01`），便于服务端做
频率限制与观测；它不参与授权。

## 3. 领取任务

```
POST /api/v1/ai/turn-claims
Authorization: Bearer <engine_token>
Content-Type: application/json

{
  "worker_id": "classical-worker-01",
  "wait_seconds": 20
}
```

### 3.1 成功（201）

```json
{
  "claim_id": "claim_abc123",
  "turn_id": "game_001:12",
  "expires_at": "2026-11-15T13:30:00Z",
  "game_id": "game_001",
  "version": 12,
  "player": 1,
  "board": [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "…15 行，每行 15 列…"],
  "history": [
    { "row": 7, "col": 7, "player": 1, "number": 1 }
  ],
  "last_move": { "row": 7, "col": 7, "player": 1, "number": 1 },
  "config": { "depth": 3 }
}
```

* `board` 是完整 15×15 二维数组，0 空位、1 黑棋、2 白棋。
* `history` 是完整落子历史，按顺序排列。
* `config` 是创建棋局时为该 AI 配置的私有参数，原样回传。
* `version` 是领取时的棋局版本；提交落子时必须原样带回。
* `player` 是本回合该 AI 执的颜色（1 黑 / 2 白）。
* `turn_id` 形如 `<game_id>:<version>`。

**领取响应已经包含计算所需的全部信息，正常情况下不需要再调用
`GET /api/v1/games/{id}`。**

### 3.2 无任务（204）

等待窗口内没有可领取的回合，返回 `204 No Content`，无响应体。
这是正常情况，不是错误。

### 3.3 领取条件

服务端只有在以下条件全部满足时才会分配任务：

1. 棋局正在进行（`status == playing`）；
2. 当前行动方是 AI；
3. 该 AI 的 `engine_id` 与 Worker 凭据匹配；
4. 对局没有暂停（`execution_state == running`）；
5. 当前回合不存在有效租约；
6. 棋局没有被删除。

同一回合同时最多存在一个有效租约。多个 Worker 并发轮询时，由服务端的原子操作
保证只有一个成功。

### 3.4 取消与超时

* 客户端取消请求或断开连接，服务端立即停止等待，不再分配任务。
* 服务端等待上限为 `wait_seconds`（0～120，默认 20）。

## 4. 续期租约

LLM AI 在长时间推理期间需要续期：

```
PATCH /api/v1/ai/turn-claims/{claim_id}
Authorization: Bearer <engine_token>
Content-Type: application/json

{ "extend_seconds": 120 }
```

响应 `200`：

```json
{ "claim_id": "claim_abc123", "expires_at": "2026-11-15T13:32:00Z" }
```

规则：

* 只有租约所属 Worker 所代表的授权引擎可以续期，其他引擎返回 `403 FORBIDDEN`。
* 单次续期长度上限 300 秒；同一租约的累计时长有总上限，超出返回
  `409 CLAIM_INVALID`。
* 租约已过期返回 `409 CLAIM_EXPIRED`。
* 建议在剩余租约不足一半时续期。

## 5. 主动释放租约

当 AI 无法完成计算（超时、模型报错、决定放弃）时主动释放：

```
DELETE /api/v1/ai/turn-claims/{claim_id}
Authorization: Bearer <engine_token>
```

响应 `204 No Content`。

释放后，若棋局仍满足领取条件，该回合重新进入待领取队列。
服务端会限制同一回合的领取频率，避免异常 Worker 无限占用 CPU 或产生大量
LLM 调用费用。

## 6. 提交落子

人类与 AI 使用**同一个**接口、同一套规则；只有身份校验不同。

```
POST /api/v1/games/{game_id}/moves
Authorization: Bearer <engine_token>
Idempotency-Key: <唯一请求标识>
Content-Type: application/json

{
  "row": 7,
  "col": 8,
  "expected_version": 12,
  "claim_id": "claim_abc123"
}
```

响应 `201`：

```json
{
  "move": { "row": 7, "col": 8, "player": 1, "number": 13 },
  "game": { "…完整棋局快照…": "…" }
}
```

### 6.1 服务端校验顺序

1. **身份**：Bearer Token 对应的引擎必须是该回合的 AI；`claim_id` 必须属于该引擎。
2. **回合**：只有当前行动方可以落子，否则 `403 NOT_YOUR_TURN`。
3. **落点**：越界 `422 POSITION_OUT_OF_RANGE`；已有棋子 `409 POSITION_OCCUPIED`。
4. **版本**：`expected_version` 与当前 `version` 不一致 → `409 VERSION_CONFLICT`。
5. **游戏状态**：已结束 → `409 GAME_FINISHED`；暂停中 → `409 GAME_PAUSED`。
6. **AI 租约**：租约不存在 → `409 CLAIM_INVALID`；已过期 → `409 CLAIM_EXPIRED`。

### 6.2 幂等

`Idempotency-Key` 请求头是**必填**的，缺失时服务端返回 `400 INVALID_REQUEST`。

`Idempotency-Key` 按「调用者 + 接口」作用域去重：

* 相同键 + 相同请求体重试 → 返回**已保存的结果**（HTTP 201），不会重复落子。
* 相同键 + 不同请求体 → `409 IDEMPOTENCY_KEY_REUSED`。

Worker 因网络问题重发时必须复用同一个 `Idempotency-Key`。

### 6.3 错误码

| HTTP | code | 含义 | Worker 建议动作 |
| --- | --- | --- | --- |
| 400 | `INVALID_REQUEST` | 请求格式错误 | 修复请求，勿重试 |
| 401 | `UNAUTHORIZED` | Token 缺失/无效 | 修复凭据，勿重试 |
| 403 | `FORBIDDEN` | Token 无权操作该棋局 | 停止该局 |
| 403 | `NOT_YOUR_TURN` | 不是你的回合 | 丢弃结果，继续轮询 |
| 404 | `GAME_NOT_FOUND` | 棋局已删除 | 丢弃结果，继续轮询 |
| 409 | `VERSION_CONFLICT` | 棋局已变化 | 丢弃过期结果，继续轮询 |
| 409 | `POSITION_OCCUPIED` | 落点已有棋子 | 重新计算 |
| 409 | `GAME_FINISHED` | 棋局已结束 | 停止该局 |
| 409 | `GAME_PAUSED` | 棋局暂停中 | 等待恢复 |
| 409 | `CLAIM_EXPIRED` | 租约过期 | 丢弃结果，继续轮询 |
| 409 | `CLAIM_INVALID` | 租约无效 | 释放并继续轮询 |
| 422 | `POSITION_OUT_OF_RANGE` | 坐标越界 | 重新计算 |
| 500 | `INTERNAL_ERROR` | 服务端错误 | 退避后重试 |

**过期任务不得被自动转换成随机落子。** 收到冲突类错误时丢弃结果即可。

## 7. 租约失效的时机

以下情况租约立即失效：

* 租约超时；
* 棋局被删除；
* 游戏结束；
* 棋局被暂停；
* Worker 主动释放；
* 该回合已经成功落子。

失效后迟到的 AI 结果必须被拒绝（`CLAIM_EXPIRED` / `CLAIM_INVALID`）。

## 8. 棋局创建与模式

创建棋局由本地界面完成，AI 不参与：

```
POST /api/v1/games
X-Game-Owner-Token 不适用（本地界面通过内部应用服务创建）
```

三种模式对 Worker 的影响：

| 模式 | Worker 行为 |
| --- | --- |
| `ai_vs_ai` | 黑、白都是 AI；每轮落子成功后自动开放下一轮 |
| `human_vs_ai` | 只有 AI 那一侧会产生任务；AI 执黑时创建后立即可领取 |
| `human_vs_human` | 不产生任何 AI 任务 |

## 9. 自检清单

接入前确认：

- [ ] Bearer Token 属于正确的 `engine_id`。
- [ ] 每次 POST 落子都带 `Idempotency-Key`，重试复用同一个键（缺失会被拒绝）。
- [ ] `expected_version` 用领取响应里的 `version`。
- [ ] `claim_id` 用领取响应里的 `claim_id`。
- [ ] 长推理期间定期 `PATCH` 续期。
- [ ] 收到 409 时丢弃结果，不做随机落子。
- [ ] 收到 204 时继续轮询，不当作错误。
