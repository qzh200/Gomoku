package contracts

import (
	"net/http"
	"testing"
)

// createGame creates a game and returns its id and owner token.
func (h *harness) createGame(mode string, players map[string]any) (string, string) {
	h.t.Helper()
	status, data := h.do(http.MethodPost, "/api/v1/games", map[string]any{"mode": mode, "players": players}, h.localHeaders())
	if status != http.StatusCreated {
		h.t.Fatalf("创建棋局失败: %d %s", status, data)
	}
	out := decode(h.t, data)
	game := out["game"].(map[string]any)
	token, _ := out["owner_token"].(string)
	return game["id"].(string), token
}

// humanVsHuman returns the players of a human game.
func humanVsHuman() map[string]any {
	return map[string]any{
		"black": map[string]any{"type": "human"},
		"white": map[string]any{"type": "human"},
	}
}

// aiPlayers returns the players of a human vs AI game.
func aiPlayers(engineID string) map[string]any {
	return map[string]any{
		"black": map[string]any{"type": "human"},
		"white": map[string]any{"type": "ai", "engine_id": engineID, "config": map[string]any{"depth": 3}},
	}
}

func TestCreateGameReturnsFullSnapshot(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("human_vs_human", humanVsHuman())
	if id == "" || owner == "" {
		t.Fatalf("创建棋局必须返回 id 与 owner_token")
	}

	status, data := h.do(http.MethodGet, "/api/v1/games/"+id, nil, nil)
	if status != http.StatusOK {
		t.Fatalf("查询棋局失败: %d %s", status, data)
	}
	game := decode(t, data)
	for _, field := range []string{"id", "mode", "players", "board", "current_player", "status", "result", "winner", "last_move", "move_count", "version", "execution_state", "ai_thinking"} {
		if _, ok := game[field]; !ok {
			t.Fatalf("棋局快照缺少字段 %q", field)
		}
	}
	board := game["board"].([]any)
	if len(board) != 15 {
		t.Fatalf("棋盘必须是 15 行, 实际 %d", len(board))
	}
	for _, row := range board {
		cells := row.([]any)
		if len(cells) != 15 {
			t.Fatalf("棋盘每行必须是 15 列, 实际 %d", len(cells))
		}
	}
	if game["current_player"].(float64) != 1 {
		t.Fatalf("黑棋先手")
	}
	if game["status"] != "playing" || game["result"] != "ongoing" || game["version"].(float64) != 0 {
		t.Fatalf("新棋局状态不对: %v", game)
	}
	if game["winner"] != nil || game["last_move"] != nil {
		t.Fatalf("新棋局没有胜者和上一步")
	}
	if game["ai_thinking"] != false {
		t.Fatalf("新棋局没有 AI 任务")
	}
}

func TestOwnerTokenIsNotExposedAgain(t *testing.T) {
	h := newHarness(t)
	id, _ := h.createGame("human_vs_human", humanVsHuman())
	status, data := h.do(http.MethodGet, "/api/v1/games/"+id, nil, nil)
	if status != http.StatusOK {
		t.Fatalf("查询棋局失败: %d %s", status, data)
	}
	if _, ok := decode(t, data)["owner_token"]; ok {
		t.Fatalf("owner_token 不得在查询接口再次返回")
	}
}

func TestCreateGameRequiresLocalClient(t *testing.T) {
	h := newHarness(t)
	status, data := h.do(http.MethodPost, "/api/v1/games", map[string]any{"mode": "human_vs_human", "players": humanVsHuman()}, nil)
	if status != http.StatusUnauthorized {
		t.Fatalf("缺少本地凭据必须返回 401, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "UNAUTHORIZED" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestCreateGameRejectsInvalidConfig(t *testing.T) {
	h := newHarness(t)
	cases := []struct {
		name    string
		mode    string
		players map[string]any
	}{
		{"未知模式", "robot_vs_robot", humanVsHuman()},
		{"AI 模式缺少引擎", "ai_vs_ai", map[string]any{
			"black": map[string]any{"type": "ai"},
			"white": map[string]any{"type": "ai", "engine_id": "classical"},
		}},
		{"未注册引擎", "human_vs_ai", aiPlayers("unknown")},
		{"模式与玩家不符", "ai_vs_ai", humanVsHuman()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			status, data := h.do(http.MethodPost, "/api/v1/games", map[string]any{"mode": tc.mode, "players": tc.players}, h.localHeaders())
			if status != http.StatusUnprocessableEntity {
				t.Fatalf("非法配置必须返回 422, 实际 %d %s", status, data)
			}
			if code := errorCode(t, data); code != "INVALID_GAME_CONFIG" {
				t.Fatalf("错误码不对: %s", code)
			}
		})
	}
}

func TestListGamesIsolatesGames(t *testing.T) {
	h := newHarness(t)
	first, _ := h.createGame("human_vs_human", humanVsHuman())
	second, _ := h.createGame("human_vs_human", humanVsHuman())

	status, data := h.do(http.MethodGet, "/api/v1/games", nil, h.localHeaders())
	if status != http.StatusOK {
		t.Fatalf("查询棋局列表失败: %d %s", status, data)
	}
	items := decode(t, data)["items"].([]any)
	if len(items) != 2 {
		t.Fatalf("棋局列表必须包含 2 个棋局, 实际 %d", len(items))
	}
	if items[0].(map[string]any)["id"] != first || items[1].(map[string]any)["id"] != second {
		t.Fatalf("棋局列表顺序不对")
	}
}

func TestGetUnknownGameReturns404(t *testing.T) {
	h := newHarness(t)
	status, data := h.do(http.MethodGet, "/api/v1/games/game_999", nil, nil)
	if status != http.StatusNotFound {
		t.Fatalf("不存在的棋局必须返回 404, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "GAME_NOT_FOUND" {
		t.Fatalf("错误码不对: %s", code)
	}
}
