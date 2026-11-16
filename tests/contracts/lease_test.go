package contracts

import (
	"net/http"
	"testing"
	"time"
)

func TestExpiredLeaseIsReassignedAfterWorkerExit(t *testing.T) {
	h := newHarnessWithLease(t, 150*time.Millisecond)
	h.createGame("ai_vs_ai", aiVsAIPlayers())

	status, data := h.claim("classical", "worker-1", 0)
	if status != http.StatusCreated {
		t.Fatalf("领取任务失败: %d %s", status, data)
	}
	claim := decode(t, data)

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		status, data = h.claim("classical", "worker-2", 0)
		if status == http.StatusCreated {
			newClaim := decode(t, data)
			if newClaim["claim_id"] == claim["claim_id"] {
				t.Fatalf("重新领取必须得到新的租约")
			}
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("租约过期后必须能重新领取")
}

func TestExpiredLeaseRejectsLateResult(t *testing.T) {
	h := newHarnessWithLease(t, 100*time.Millisecond)
	id, _ := h.createGame("ai_vs_ai", aiVsAIPlayers())

	status, data := h.claim("classical", "worker-1", 0)
	if status != http.StatusCreated {
		t.Fatalf("领取任务失败: %d %s", status, data)
	}
	claimID := decode(t, data)["claim_id"].(string)

	time.Sleep(200 * time.Millisecond)

	status, data = h.do(http.MethodPost, "/api/v1/games/"+id+"/moves", map[string]any{
		"row": 7, "col": 7, "expected_version": 0, "claim_id": claimID,
	}, engineHeaders("classical"))
	if status != http.StatusConflict {
		t.Fatalf("过期的结果必须被拒绝, 实际 %d %s", status, data)
	}
	code := errorCode(t, data)
	if code != "CLAIM_EXPIRED" && code != "CLAIM_INVALID" {
		t.Fatalf("错误码不对: %s", code)
	}
}

func TestLongThinkingWorkerRenewsItsLease(t *testing.T) {
	h := newHarnessWithLease(t, 200*time.Millisecond)
	id, _ := h.createGame("ai_vs_ai", map[string]any{
		"black": map[string]any{"type": "ai", "engine_id": "llm"},
		"white": map[string]any{"type": "ai", "engine_id": "classical"},
	})

	status, data := h.claim("llm", "llm-worker-1", 0)
	if status != http.StatusCreated {
		t.Fatalf("领取任务失败: %d %s", status, data)
	}
	claimID := decode(t, data)["claim_id"].(string)

	for range 3 {
		time.Sleep(100 * time.Millisecond)
		status, data = h.do(http.MethodPatch, "/api/v1/ai/turn-claims/"+claimID, map[string]any{"extend_seconds": 60}, engineHeaders("llm"))
		if status != http.StatusOK {
			t.Fatalf("续期失败: %d %s", status, data)
		}
	}

	status, data = h.do(http.MethodPost, "/api/v1/games/"+id+"/moves", map[string]any{
		"row": 7, "col": 7, "expected_version": 0, "claim_id": claimID,
	}, engineHeaders("llm"))
	if status != http.StatusCreated {
		t.Fatalf("续期后的落子必须成功, 实际 %d %s", status, data)
	}
}
