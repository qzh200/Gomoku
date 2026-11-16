package scheduler

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"
)

const (
	MaxExtendSeconds = 300

	MaxLeaseTotalSeconds = 900

	DefaultLeaseSeconds = 60

	RetryBackoff = 2 * time.Second
)

var (
	ErrClaimNotFound = errors.New("claim not found")
	ErrClaimExpired  = errors.New("claim expired")
	ErrClaimInvalid  = errors.New("claim invalid")
	ErrForbidden     = errors.New("claim belongs to another engine")
	ErrLeaseTooLong  = errors.New("lease lifetime exhausted")
)

type Lease struct {
	ClaimID   string
	TurnID    string
	GameID    string
	EngineID  string
	WorkerID  string
	Version   int
	Player    int
	StartsAt  time.Time
	ExpiresAt time.Time
}

func (l *Lease) Expired(now time.Time) bool { return !now.Before(l.ExpiresAt) }

type LeaseManager struct {
	mu       sync.Mutex
	leases   map[string]*Lease
	byTurn   map[string]string
	retryAt  map[string]time.Time
	attempts map[string]int
}

func NewLeaseManager() *LeaseManager {
	return &LeaseManager{
		leases:   make(map[string]*Lease),
		byTurn:   make(map[string]string),
		retryAt:  make(map[string]time.Time),
		attempts: make(map[string]int),
	}
}

func TurnID(gameID string, version int) string { return fmt.Sprintf("%s:%d", gameID, version) }

func (m *LeaseManager) TryAcquire(turnID, gameID, engineID, workerID string, version, player int, ttl time.Duration) (*Lease, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	if existing, ok := m.byTurn[turnID]; ok {
		if lease, ok := m.leases[existing]; ok {
			if !lease.Expired(now) {
				return nil, false
			}
			delete(m.leases, existing)
		}
		delete(m.byTurn, turnID)
	}
	if until, ok := m.retryAt[turnID]; ok && now.Before(until) {
		return nil, false
	}
	if ttl <= 0 {
		ttl = DefaultLeaseSeconds * time.Second
	}
	lease := &Lease{
		ClaimID:   newClaimID(),
		TurnID:    turnID,
		GameID:    gameID,
		EngineID:  engineID,
		WorkerID:  workerID,
		Version:   version,
		Player:    player,
		StartsAt:  now,
		ExpiresAt: now.Add(ttl),
	}
	m.leases[lease.ClaimID] = lease
	m.byTurn[turnID] = lease.ClaimID
	m.attempts[turnID]++
	return lease, true
}

func (m *LeaseManager) Get(claimID string) (*Lease, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	lease, ok := m.leases[claimID]
	if !ok {
		return nil, false
	}
	copied := *lease
	return &copied, true
}

func (m *LeaseManager) Renew(claimID, engineID string, extendSeconds int) (*Lease, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	lease, ok := m.leases[claimID]
	if !ok {
		return nil, ErrClaimNotFound
	}
	if lease.EngineID != engineID {
		return nil, ErrForbidden
	}
	if lease.Expired(now) {
		delete(m.leases, claimID)
		delete(m.byTurn, lease.TurnID)
		return nil, ErrClaimExpired
	}
	if extendSeconds <= 0 || extendSeconds > MaxExtendSeconds {
		return nil, ErrClaimInvalid
	}
	total := now.Add(time.Duration(extendSeconds) * time.Second).Sub(lease.StartsAt)
	if total > MaxLeaseTotalSeconds*time.Second {
		return nil, ErrLeaseTooLong
	}
	lease.ExpiresAt = now.Add(time.Duration(extendSeconds) * time.Second)
	copied := *lease
	return &copied, nil
}

func (m *LeaseManager) Release(claimID, engineID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	lease, ok := m.leases[claimID]
	if !ok {
		return ErrClaimNotFound
	}
	if lease.EngineID != engineID {
		return ErrForbidden
	}
	delete(m.leases, claimID)
	delete(m.byTurn, lease.TurnID)
	m.retryAt[lease.TurnID] = time.Now().Add(RetryBackoff)
	return nil
}

func (m *LeaseManager) ActiveFor(gameID string, version int) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	claimID, ok := m.byTurn[TurnID(gameID, version)]
	if !ok {
		return false
	}
	lease, ok := m.leases[claimID]
	if !ok {
		return false
	}
	if lease.Expired(now) {
		delete(m.leases, claimID)
		delete(m.byTurn, lease.TurnID)
		return false
	}
	return true
}

func (m *LeaseManager) InvalidateTurn(turnID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	claimID, ok := m.byTurn[turnID]
	if !ok {
		return
	}
	delete(m.leases, claimID)
	delete(m.byTurn, turnID)
}

func (m *LeaseManager) InvalidateGame(gameID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for claimID, lease := range m.leases {
		if lease.GameID == gameID {
			delete(m.leases, claimID)
			delete(m.byTurn, lease.TurnID)
		}
	}
}

func newClaimID() string {
	buf := make([]byte, 12)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("claim_%d", time.Now().UnixNano())
	}
	return "claim_" + hex.EncodeToString(buf)
}
