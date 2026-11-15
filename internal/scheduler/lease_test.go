package scheduler

import (
	"testing"
	"time"
)

func TestOnlyOneLeasePerTurn(t *testing.T) {
	m := NewLeaseManager()
	first, ok := m.TryAcquire("game_001:0", "game_001", "classical", "worker-1", 0, 1, time.Minute)
	if !ok {
		t.Fatalf("第一次领取必须成功")
	}
	if _, ok := m.TryAcquire("game_001:0", "game_001", "classical", "worker-2", 0, 1, time.Minute); ok {
		t.Fatalf("同一回合只能有一个租约")
	}
	if !m.ActiveFor("game_001", 0) {
		t.Fatalf("当前回合必须有有效租约")
	}
	if first.ClaimID == "" {
		t.Fatalf("租约必须有标识")
	}
}

func TestExpiredLeaseIsReassigned(t *testing.T) {
	m := NewLeaseManager()
	if _, ok := m.TryAcquire("game_001:0", "game_001", "classical", "worker-1", 0, 1, time.Millisecond); !ok {
		t.Fatalf("第一次领取必须成功")
	}
	time.Sleep(5 * time.Millisecond)
	if m.ActiveFor("game_001", 0) {
		t.Fatalf("过期租约不得有效")
	}
	if _, ok := m.TryAcquire("game_001:0", "game_001", "classical", "worker-2", 0, 1, time.Minute); !ok {
		t.Fatalf("过期租约后必须能重新领取")
	}
}

func TestRenewOnlyByOwner(t *testing.T) {
	m := NewLeaseManager()
	lease, _ := m.TryAcquire("game_001:0", "game_001", "classical", "worker-1", 0, 1, time.Minute)
	before := lease.ExpiresAt
	if _, err := m.Renew(lease.ClaimID, "llm", 60); err != ErrForbidden {
		t.Fatalf("其他引擎续期必须被拒绝, 实际 %v", err)
	}
	renewed, err := m.Renew(lease.ClaimID, "classical", 120)
	if err != nil {
		t.Fatalf("续期必须成功: %v", err)
	}
	if !renewed.ExpiresAt.After(before) {
		t.Fatalf("续期必须延后到期时间")
	}
}

func TestRenewRejectsExcessiveExtension(t *testing.T) {
	m := NewLeaseManager()
	lease, _ := m.TryAcquire("game_001:0", "game_001", "classical", "worker-1", 0, 1, time.Minute)
	if _, err := m.Renew(lease.ClaimID, "classical", MaxExtendSeconds+1); err != ErrClaimInvalid {
		t.Fatalf("过长的续期必须被拒绝, 实际 %v", err)
	}
	if _, err := m.Renew(lease.ClaimID, "classical", 0); err != ErrClaimInvalid {
		t.Fatalf("非法的续期长度必须被拒绝, 实际 %v", err)
	}
}

func TestRenewRejectsExpiredLease(t *testing.T) {
	m := NewLeaseManager()
	lease, _ := m.TryAcquire("game_001:0", "game_001", "classical", "worker-1", 0, 1, time.Millisecond)
	time.Sleep(5 * time.Millisecond)
	if _, err := m.Renew(lease.ClaimID, "classical", 60); err != ErrClaimExpired {
		t.Fatalf("过期租约续期必须返回过期, 实际 %v", err)
	}
}

func TestReleasePutsTurnInBackoff(t *testing.T) {
	m := NewLeaseManager()
	lease, _ := m.TryAcquire("game_001:0", "game_001", "classical", "worker-1", 0, 1, time.Minute)
	if err := m.Release(lease.ClaimID, "classical"); err != nil {
		t.Fatalf("释放必须成功: %v", err)
	}
	if _, ok := m.TryAcquire("game_001:0", "game_001", "classical", "worker-2", 0, 1, time.Minute); ok {
		t.Fatalf("释放后必须进入退避, 不得立即重新领取")
	}
}

func TestInvalidateGameDropsLeases(t *testing.T) {
	m := NewLeaseManager()
	lease, _ := m.TryAcquire("game_001:0", "game_001", "classical", "worker-1", 0, 1, time.Minute)
	m.InvalidateGame("game_001")
	if m.ActiveFor("game_001", 0) {
		t.Fatalf("棋局失效后租约不得有效")
	}
	if _, ok := m.Get(lease.ClaimID); ok {
		t.Fatalf("失效的租约必须被删除")
	}
}

func TestInvalidateTurnKeepsOtherTurns(t *testing.T) {
	m := NewLeaseManager()
	m.TryAcquire("game_001:0", "game_001", "classical", "worker-1", 0, 1, time.Minute)
	m.TryAcquire("game_002:0", "game_002", "classical", "worker-2", 0, 1, time.Minute)
	m.InvalidateTurn("game_001:0")
	if m.ActiveFor("game_001", 0) {
		t.Fatalf("被失效的回合不得有有效租约")
	}
	if !m.ActiveFor("game_002", 0) {
		t.Fatalf("其他回合的租约不得受影响")
	}
}
