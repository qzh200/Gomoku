package scheduler

import (
	"testing"
	"time"
)

func TestQueueLeasesTheNextTurn(t *testing.T) {
	leases := NewLeaseManager()
	queue := NewQueue(leases, func(engineID string) []Turn {
		return []Turn{{GameID: "game_001", Version: 0, Player: 1, EngineID: engineID, Config: map[string]any{"depth": 3}}}
	}, time.Minute)

	lease, turn, ok := queue.Next("classical", "worker-1")
	if !ok {
		t.Fatalf("必须领取到回合")
	}
	if lease.GameID != "game_001" || lease.Version != 0 || lease.Player != 1 {
		t.Fatalf("租约内容不对: %+v", lease)
	}
	if turn.Config["depth"] != 3 {
		t.Fatalf("引擎配置必须随回合返回: %v", turn.Config)
	}
	if _, _, ok := queue.Next("classical", "worker-2"); ok {
		t.Fatalf("同一回合不得被领取两次")
	}
}

func TestQueueSkipsTurnsOfOtherEngines(t *testing.T) {
	leases := NewLeaseManager()
	queue := NewQueue(leases, func(engineID string) []Turn {
		if engineID != "classical" {
			return nil
		}
		return []Turn{{GameID: "game_001", Version: 0, Player: 1, EngineID: engineID}}
	}, time.Minute)

	if _, _, ok := queue.Next("llm", "worker-1"); ok {
		t.Fatalf("不得领取其他引擎的回合")
	}
	if _, _, ok := queue.Next("classical", "worker-1"); !ok {
		t.Fatalf("必须领取本引擎的回合")
	}
}

func TestQueueWithoutTurnReportsNothing(t *testing.T) {
	leases := NewLeaseManager()
	queue := NewQueue(leases, func(string) []Turn { return nil }, time.Minute)
	if _, _, ok := queue.Next("classical", "worker-1"); ok {
		t.Fatalf("没有回合时不得领取")
	}
}
