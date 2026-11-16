package contracts

import (
	"net/http"
	"sync"
	"testing"
)

func aiVsAIPlayers() map[string]any {
	return map[string]any{
		"black": map[string]any{"type": "ai", "engine_id": "classical"},
		"white": map[string]any{"type": "ai", "engine_id": "llm"},
	}
}

func (h *harness) claim(engineID, workerID string, waitSeconds int) (int, []byte) {
	h.t.Helper()
	return h.do(http.MethodPost, "/api/v1/ai/turn-claims", map[string]any{
		"worker_id":    workerID,
		"wait_seconds": waitSeconds,
	}, engineHeaders(engineID))
}

func TestClaimRequiresEngineCredential(t *testing.T) {
	h := newHarness(t)
	status, data := h.do(http.MethodPost, "/api/v1/ai/turn-claims", map[string]any{"worker_id": "w1"}, nil)
	if status != http.StatusUnauthorized {
		t.Fatalf("缺少引擎凭据必须返回 401, 实际 %d %s", status, data)
	}
}

func TestClaimWithoutTaskReturns204(t *testing.T) {
	h := newHarness(t)
	status, data := h.claim("classical", "worker-1", 0)
	if status != http.StatusNoContent {
		t.Fatalf("没有任务必须返回 204, 实际 %d %s", status, data)
	}
	if len(data) != 0 {
		t.Fatalf("204 不得有响应体: %s", data)
	}
}

func TestClaimReturnsFullTask(t *testing.T) {
	h := newHarness(t)
	id, _ := h.createGame("ai_vs_ai", aiVsAIPlayers())
	status, data := h.claim("classical", "worker-1", 0)
	if status != http.StatusCreated {
		t.Fatalf("领取任务必须返回 201, 实际 %d %s", status, data)
	}
	claim := decode(t, data)
	for _, field := range []string{"claim_id", "turn_id", "expires_at", "game_id", "version", "player", "board", "history", "last_move", "config"} {
		if _, ok := claim[field]; !ok {
			t.Fatalf("任务缺少字段 %q", field)
		}
	}
	if claim["game_id"] != id {
		t.Fatalf("任务指向错误的棋局: %v", claim["game_id"])
	}
	if claim["version"].(float64) != 0 || claim["player"].(float64) != 1 {
		t.Fatalf("任务版本与执子方不对: %v", claim)
	}
	if claim["turn_id"] != id+":0" {
		t.Fatalf("turn_id 格式不对: %v", claim["turn_id"])
	}
	board := claim["board"].([]any)
	if len(board) != 15 {
		t.Fatalf("任务棋盘必须是 15 行, 实际 %d", len(board))
	}
	if claim["last_move"] != nil {
		t.Fatalf("第一回合没有上一步")
	}
}

func TestOnlyTheMatchingEngineClaimsTheTurn(t *testing.T) {
	h := newHarness(t)
	h.createGame("ai_vs_ai", aiVsAIPlayers())
	status, data := h.claim("llm", "worker-llm", 0)
	if status != http.StatusNoContent {
		t.Fatalf("引擎不匹配时不得领取任务, 实际 %d %s", status, data)
	}
}

func TestSecondWorkerCannotClaimTheSameTurn(t *testing.T) {
	h := newHarness(t)
	h.createGame("ai_vs_ai", aiVsAIPlayers())
	status, _ := h.claim("classical", "worker-1", 0)
	if status != http.StatusCreated {
		t.Fatalf("第一个 Worker 必须领取成功")
	}
	status, data := h.claim("classical", "worker-2", 0)
	if status != http.StatusNoContent {
		t.Fatalf("同一回合只能有一个租约, 实际 %d %s", status, data)
	}
}

func TestConcurrentWorkersOnlyOneWins(t *testing.T) {
	h := newHarness(t)
	h.createGame("ai_vs_ai", aiVsAIPlayers())

	const workers = 8
	var wg sync.WaitGroup
	var mu sync.Mutex
	won := 0
	for i := range workers {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			status, _ := h.claim("classical", "worker-"+string(rune('a'+worker)), 2)
			if status == http.StatusCreated {
				mu.Lock()
				won++
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	if won != 1 {
		t.Fatalf("并发领取时只有一个 Worker 能成功, 实际 %d", won)
	}
}

func TestAIMoveRequiresClaim(t *testing.T) {
	h := newHarness(t)
	id, _ := h.createGame("ai_vs_ai", aiVsAIPlayers())
	status, data := h.do(http.MethodPost, "/api/v1/games/"+id+"/moves", map[string]any{
		"row": 7, "col": 7, "expected_version": 0,
	}, engineHeaders("classical"))
	if status != http.StatusConflict {
		t.Fatalf("没有租约的 AI 落子必须返回 409, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "CLAIM_INVALID" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestAIMoveWithClaimSucceeds(t *testing.T) {
	h := newHarness(t)
	id, _ := h.createGame("ai_vs_ai", aiVsAIPlayers())
	status, data := h.claim("classical", "worker-1", 0)
	if status != http.StatusCreated {
		t.Fatalf("领取任务失败: %d %s", status, data)
	}
	claim := decode(t, data)
	status, data = h.do(http.MethodPost, "/api/v1/games/"+id+"/moves", map[string]any{
		"row": 7, "col": 7, "expected_version": 0, "claim_id": claim["claim_id"],
	}, engineHeaders("classical"))
	if status != http.StatusCreated {
		t.Fatalf("AI 落子必须成功, 实际 %d %s", status, data)
	}
	game := decode(t, data)["game"].(map[string]any)
	if game["version"].(float64) != 1 {
		t.Fatalf("AI 落子后版本必须是 1: %v", game["version"])
	}
}

func TestRenewAndReleaseClaim(t *testing.T) {
	h := newHarness(t)
	h.createGame("ai_vs_ai", aiVsAIPlayers())
	status, data := h.claim("classical", "worker-1", 0)
	if status != http.StatusCreated {
		t.Fatalf("领取任务失败: %d %s", status, data)
	}
	claimID := decode(t, data)["claim_id"].(string)

	status, data = h.do(http.MethodPatch, "/api/v1/ai/turn-claims/"+claimID, map[string]any{"extend_seconds": 120}, engineHeaders("classical"))
	if status != http.StatusOK {
		t.Fatalf("续期必须返回 200, 实际 %d %s", status, data)
	}
	renewed := decode(t, data)
	if renewed["claim_id"] != claimID || renewed["expires_at"] == nil {
		t.Fatalf("续期响应不对: %v", renewed)
	}

	status, data = h.do(http.MethodPatch, "/api/v1/ai/turn-claims/"+claimID, map[string]any{"extend_seconds": 60}, engineHeaders("llm"))
	if status != http.StatusForbidden {
		t.Fatalf("其他引擎续期必须返回 403, 实际 %d %s", status, data)
	}

	status, data = h.do(http.MethodDelete, "/api/v1/ai/turn-claims/"+claimID, nil, engineHeaders("classical"))
	if status != http.StatusNoContent {
		t.Fatalf("释放必须返回 204, 实际 %d %s", status, data)
	}
}

func TestReleaseUnknownClaimReturns404(t *testing.T) {
	h := newHarness(t)
	status, data := h.do(http.MethodDelete, "/api/v1/ai/turn-claims/claim_missing", nil, engineHeaders("classical"))
	if status != http.StatusNotFound {
		t.Fatalf("不存在的租约必须返回 404, 实际 %d %s", status, data)
	}
	if code := errorCode(t, data); code != "CLAIM_NOT_FOUND" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestPauseInvalidatesLease(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("ai_vs_ai", aiVsAIPlayers())
	status, data := h.claim("classical", "worker-1", 0)
	if status != http.StatusCreated {
		t.Fatalf("领取任务失败: %d %s", status, data)
	}
	claimID := decode(t, data)["claim_id"].(string)

	status, data = h.do(http.MethodPatch, "/api/v1/games/"+id, map[string]any{"execution_state": "paused"}, ownerHeaders(owner))
	if status != http.StatusOK {
		t.Fatalf("暂停必须返回 200, 实际 %d %s", status, data)
	}

	status, data = h.do(http.MethodPost, "/api/v1/games/"+id+"/moves", map[string]any{
		"row": 7, "col": 7, "expected_version": 0, "claim_id": claimID,
	}, engineHeaders("classical"))
	if status != http.StatusConflict {
		t.Fatalf("暂停后的 AI 落子必须返回 409, 实际 %d %s", status, data)
	}

	status, data = h.claim("classical", "worker-2", 0)
	if status != http.StatusNoContent {
		t.Fatalf("暂停后不得领取任务, 实际 %d %s", status, data)
	}
}

func TestResumeReopensTheTurn(t *testing.T) {
	h := newHarness(t)
	id, owner := h.createGame("ai_vs_ai", aiVsAIPlayers())
	if status, data := h.do(http.MethodPatch, "/api/v1/games/"+id, map[string]any{"execution_state": "paused"}, ownerHeaders(owner)); status != http.StatusOK {
		t.Fatalf("暂停失败: %d %s", status, data)
	}
	if status, data := h.do(http.MethodPatch, "/api/v1/games/"+id, map[string]any{"execution_state": "running"}, ownerHeaders(owner)); status != http.StatusOK {
		t.Fatalf("恢复失败: %d %s", status, data)
	}
	status, data := h.claim("classical", "worker-2", 0)
	if status != http.StatusCreated {
		t.Fatalf("恢复后必须可以领取任务, 实际 %d %s", status, data)
	}
}
