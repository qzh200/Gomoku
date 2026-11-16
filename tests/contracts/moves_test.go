package contracts

import (
	"net/http"
	"testing"
)

func (h *harness) move(id string, row, col, version int, headers map[string]string) (int, []byte) {
	h.t.Helper()
	body := map[string]any{"row": row, "col": col, "expected_version": version}
	return h.do(http.MethodPost, "/api/v1/games/"+id+"/moves", body, headers)
}

func TestHumanMoveUpdatesGame(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())

	status, data := h.move(id, 7, 7, 0, ownerHeaders(owner))
	if status != http.StatusCreated {
		t.Fatalf("落子必须返回 201, 实际 %d %s", status, data)
	}
	out := decode(t, data)
	move := out["move"].(map[string]any)
	if move["row"].(float64) != 7 || move["col"].(float64) != 7 || move["player"].(float64) != 1 || move["number"].(float64) != 1 {
		t.Fatalf("落子记录不对: %v", move)
	}
	game := out["game"].(map[string]any)
	if game["version"].(float64) != 1 || game["move_count"].(float64) != 1 {
		t.Fatalf("版本与手数不对: %v", game)
	}
	if game["current_player"].(float64) != 2 {
		t.Fatalf("回合必须交给白棋")
	}
	if game["last_move"].(map[string]any)["number"].(float64) != 1 {
		t.Fatalf("last_move 必须是最后一步")
	}
}

func TestMoveHistoryKeepsOrder(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	for i := range 3 {
		status, data := h.move(id, 7, 7+i, i, ownerHeaders(owner))
		if status != http.StatusCreated {
			t.Fatalf("第 %d 手失败: %d %s", i, status, data)
		}
	}
	status, data := h.do(http.MethodGet, "/api/v1/games/"+id+"/moves", nil, nil)
	if status != http.StatusOK {
		t.Fatalf("查询历史失败: %d %s", status, data)
	}
	items := decode(t, data)["items"].([]any)
	if len(items) != 3 {
		t.Fatalf("历史必须有 3 手, 实际 %d", len(items))
	}
	for i, item := range items {
		move := item.(map[string]any)
		if move["number"].(float64) != float64(i+1) {
			t.Fatalf("历史顺序不对: %v", move)
		}
	}
}

func TestMoveRejectsWrongTurn(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_ai", aiPlayers("classical"))

	if status, data := h.move(id, 7, 7, 0, ownerHeaders(owner)); status != http.StatusCreated {
		t.Fatalf("人类第一手必须成功: %d %s", status, data)
	}

	status, data := h.move(id, 8, 8, 1, ownerHeaders(owner))
	if status != http.StatusForbidden {
		t.Fatalf("错误回合必须返回 403, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "NOT_YOUR_TURN" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestMoveRejectsOccupiedPosition(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	if status, data := h.move(id, 7, 7, 0, ownerHeaders(owner)); status != http.StatusCreated {
		t.Fatalf("第一手失败: %d %s", status, data)
	}
	status, data := h.move(id, 7, 7, 1, ownerHeaders(owner))
	if status != http.StatusConflict {
		t.Fatalf("占用位置必须返回 409, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "POSITION_OCCUPIED" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestMoveRejectsOutOfRange(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	status, data := h.move(id, 15, 0, 0, ownerHeaders(owner))
	if status != http.StatusUnprocessableEntity {
		t.Fatalf("越界落点必须返回 422, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "POSITION_OUT_OF_RANGE" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestMoveRejectsWrongVersion(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	status, data := h.move(id, 7, 7, 3, ownerHeaders(owner))
	if status != http.StatusConflict {
		t.Fatalf("版本冲突必须返回 409, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "VERSION_CONFLICT" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestMoveRejectsForeignOwner(t *testing.T) {
	h := newHarness(t)
	id, _ := h.createGame("human_vs_human", humanVsHuman())
	status, data := h.move(id, 7, 7, 0, ownerHeaders("some-other-token"))
	if status != http.StatusForbidden {
		t.Fatalf("越权落子必须返回 403, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "FORBIDDEN" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestMoveRequiresCredential(t *testing.T) {
	h := newHarness(t)
	id, _ := h.createGame("human_vs_human", humanVsHuman())
	status, data := h.move(id, 7, 7, 0, nil)
	if status != http.StatusUnauthorized {
		t.Fatalf("缺少凭据必须返回 401, 实际 %d %s", status, data)
	}
}

func TestIdempotentMoveReturnsSavedResult(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	headers := ownerHeaders(owner)
	headers["Idempotency-Key"] = "key-1"

	status, first := h.move(id, 7, 7, 0, headers)
	if status != http.StatusCreated {
		t.Fatalf("第一手失败: %d %s", status, first)
	}
	status, second := h.move(id, 7, 7, 0, headers)
	if status != http.StatusCreated {
		t.Fatalf("重试必须返回 201, 实际 %d %s", status, second)
	}
	if decode(t, first)["move"].(map[string]any)["number"].(float64) != 1 {
		t.Fatalf("第一手必须是第 1 手")
	}
	if decode(t, second)["move"].(map[string]any)["number"].(float64) != 1 {
		t.Fatalf("重试必须返回已保存的结果")
	}

	status, data := h.do(http.MethodGet, "/api/v1/games/"+id, nil, nil)
	if status != http.StatusOK {
		t.Fatalf("查询棋局失败: %d %s", status, data)
	}
	game := decode(t, data)
	if game["move_count"].(float64) != 1 {
		t.Fatalf("重试不得重复落子: %v", game["move_count"])
	}
}

func TestIdempotencyKeyWithDifferentBodyConflicts(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	headers := ownerHeaders(owner)
	headers["Idempotency-Key"] = "key-1"
	if status, data := h.move(id, 7, 7, 0, headers); status != http.StatusCreated {
		t.Fatalf("第一手失败: %d %s", status, data)
	}
	status, data := h.move(id, 8, 8, 1, headers)
	if status != http.StatusConflict {
		t.Fatalf("同键不同请求必须返回 409, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "IDEMPOTENCY_KEY_REUSED" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestGameFinishesAfterFiveInARow(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())

	for i := range 9 {
		row, col := 7, 7+i/2
		if i%2 == 1 {
			row = 9
		}
		status, data := h.move(id, row, col, i, ownerHeaders(owner))
		if status != http.StatusCreated {
			t.Fatalf("第 %d 手失败: %d %s", i, status, data)
		}
	}
	status, data := h.do(http.MethodGet, "/api/v1/games/"+id, nil, nil)
	if status != http.StatusOK {
		t.Fatalf("查询棋局失败: %d %s", status, data)
	}
	game := decode(t, data)
	if game["status"] != "finished" || game["result"] != "black_win" {
		t.Fatalf("黑棋五连必须结束棋局: %v %v", game["status"], game["result"])
	}
	if game["winner"].(float64) != 1 {
		t.Fatalf("胜者必须是黑棋")
	}

	status, data = h.move(id, 11, 11, 9, ownerHeaders(owner))
	if status != http.StatusConflict {
		t.Fatalf("终局后落子必须返回 409, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "GAME_FINISHED" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestDeleteGame(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	status, data := h.do(http.MethodDelete, "/api/v1/games/"+id, nil, ownerHeaders(owner))
	if status != http.StatusNoContent {
		t.Fatalf("删除棋局必须返回 204, 实际 %d %s", status, data)
	}
	status, data = h.do(http.MethodGet, "/api/v1/games/"+id, nil, nil)
	if status != http.StatusNotFound {
		t.Fatalf("删除后查询必须返回 404, 实际 %d %s", status, data)
	}
}

func TestPauseOnlyForAIVsAI(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	status, data := h.do(http.MethodPatch, "/api/v1/games/"+id, map[string]any{"execution_state": "paused"}, ownerHeaders(owner))
	if status != http.StatusUnprocessableEntity {
		t.Fatalf("人人对战不能暂停, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "INVALID_GAME_CONFIG" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestPauseKeepsVersion(t *testing.T) {
	h := newHarness(t)
	players := map[string]any{
		"black": map[string]any{"type": "ai", "engine_id": "classical"},
		"white": map[string]any{"type": "ai", "engine_id": "llm"},
	}
	id, owner := h.createGame("ai_vs_ai", players)
	status, data := h.do(http.MethodPatch, "/api/v1/games/"+id, map[string]any{"execution_state": "paused", "expected_version": 0}, ownerHeaders(owner))
	if status != http.StatusOK {
		t.Fatalf("暂停必须返回 200, 实际 %d %s", status, data)
	}
	game := decode(t, data)
	if game["execution_state"] != "paused" {
		t.Fatalf("执行状态不对: %v", game["execution_state"])
	}
	if game["version"].(float64) != 0 {
		t.Fatalf("暂停不得改变棋盘版本")
	}
}

func TestEnginesEndpoint(t *testing.T) {
	h := newHarness(t)
	status, data := h.do(http.MethodGet, "/api/v1/engines", nil, nil)
	if status != http.StatusOK {
		t.Fatalf("查询引擎失败: %d %s", status, data)
	}
	items := decode(t, data)["items"].([]any)
	if len(items) != 2 {
		t.Fatalf("必须有 2 个引擎, 实际 %d", len(items))
	}
	engine := items[0].(map[string]any)
	for _, field := range []string{"engine_id", "display_name", "available", "last_active_at"} {
		if _, ok := engine[field]; !ok {
			t.Fatalf("引擎缺少字段 %q", field)
		}
	}
	if _, ok := engine["token"]; ok {
		t.Fatalf("引擎响应不得包含凭据")
	}
}

func TestErrorEnvelopeShape(t *testing.T) {
	h := newHarness(t)
	status, data := h.do(http.MethodGet, "/api/v1/games/game_404", nil, nil)
	if status != http.StatusNotFound {
		t.Fatalf("必须返回 404, 实际 %d", status)
	}
	out := decode(t, data)
	errObj := out["error"].(map[string]any)
	if errObj["code"] == "" || errObj["message"] == "" {
		t.Fatalf("错误格式不对: %v", out)
	}
	if len(out) != 1 {
		t.Fatalf("错误响应只能有 error 字段: %v", out)
	}
}

func TestHumanVsAIGameStartsWithHumanTurn(t *testing.T) {
	h := newHarness(t)
	id, _ := h.createGame("human_vs_ai", aiPlayers("classical"))
	status, data := h.do(http.MethodGet, "/api/v1/games/"+id, nil, nil)
	if status != http.StatusOK {
		t.Fatalf("查询棋局失败: %d %s", status, data)
	}
	game := decode(t, data)
	if game["current_player"].(float64) != 1 {
		t.Fatalf("人类执黑时黑棋先手")
	}
	players := game["players"].(map[string]any)
	white := players["white"].(map[string]any)
	if white["engine_id"] != "classical" {
		t.Fatalf("白棋引擎不对: %v", white)
	}
	if white["config"].(map[string]any)["depth"].(float64) != 3 {
		t.Fatalf("引擎配置必须原样保留: %v", white)
	}
	black := players["black"].(map[string]any)
	if black["engine_id"] != nil {
		t.Fatalf("人类玩家的 engine_id 必须是 null: %v", black)
	}
	_ = id
}
