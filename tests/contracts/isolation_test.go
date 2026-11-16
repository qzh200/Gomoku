package contracts

import (
	"net/http"
	"testing"
)

func TestGamesAreIsolated(t *testing.T) {
	h := newHarness(t)
	first, firstOwner := h.createGame("human_vs_human", humanVsHuman())
	second, secondOwner := h.createGame("human_vs_human", humanVsHuman())

	if status, data := h.move(first, 7, 7, 0, ownerHeaders(firstOwner)); status != http.StatusCreated {
		t.Fatalf("第一局落子失败: %d %s", status, data)
	}

	status, data := h.do(http.MethodGet, "/api/v1/games/"+second, nil, nil)
	if status != http.StatusOK {
		t.Fatalf("查询棋局失败: %d %s", status, data)
	}
	game := decode(t, data)
	if game["version"].(float64) != 0 || game["move_count"].(float64) != 0 {
		t.Fatalf("第二局不得受第一局影响: %v", game)
	}

	status, data = h.move(second, 3, 3, 0, ownerHeaders(firstOwner))
	if status != http.StatusForbidden {
		t.Fatalf("跨棋局落子必须返回 403, 实际 %d %s", status, data)
	}

	firstHeaders := ownerHeaders(firstOwner)
	firstHeaders["Idempotency-Key"] = "shared-key"
	if status, data := h.move(first, 3, 3, 1, firstHeaders); status != http.StatusCreated {
		t.Fatalf("第一局落子失败: %d %s", status, data)
	}
	secondHeaders := ownerHeaders(secondOwner)
	secondHeaders["Idempotency-Key"] = "shared-key"
	if status, data := h.move(second, 3, 3, 0, secondHeaders); status != http.StatusCreated {
		t.Fatalf("第二局落子失败: %d %s", status, data)
	}
}
