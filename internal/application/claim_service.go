package application

import (
	"context"
	"time"

	"mygo/internal/domain/gomoku"
	"mygo/internal/scheduler"
)

type Claim struct {
	ClaimID   string         `json:"claim_id"`
	TurnID    string         `json:"turn_id"`
	ExpiresAt time.Time      `json:"expires_at"`
	GameID    string         `json:"game_id"`
	Version   int            `json:"version"`
	Player    int            `json:"player"`
	Board     [][]int        `json:"board"`
	History   []gomoku.Move  `json:"history"`
	LastMove  *gomoku.Move   `json:"last_move"`
	Config    map[string]any `json:"config"`
}

type ClaimService struct {
	coord *Coordinator
	queue *scheduler.Queue
	ttl   time.Duration
}

type ClaimServiceOption func(*ClaimService)

func LeaseTTL(ttl time.Duration) ClaimServiceOption {
	return func(s *ClaimService) { s.ttl = ttl }
}

func NewClaimService(coord *Coordinator, options ...ClaimServiceOption) *ClaimService {
	service := &ClaimService{coord: coord, ttl: scheduler.DefaultLeaseSeconds * time.Second}
	for _, option := range options {
		option(service)
	}
	service.queue = scheduler.NewQueue(coord.Leases(), coord.pendingTurns, service.ttl)
	return service
}

func (s *ClaimService) Claim(ctx context.Context, engineID, workerID string, wait time.Duration) (*Claim, bool, error) {
	deadline := time.Now().Add(wait)
	for {
		claim, ok, err := s.tryClaim(engineID, workerID)
		if err != nil {
			return nil, false, err
		}
		if ok {
			return claim, true, nil
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return nil, false, nil
		}
		if err := s.wait(ctx, remaining); err != nil {
			return nil, false, err
		}
	}
}

func (s *ClaimService) tryClaim(engineID, workerID string) (*Claim, bool, error) {
	s.coord.Lock()
	defer s.coord.Unlock()

	lease, turn, ok := s.queue.Next(engineID, workerID)
	if !ok {
		return nil, false, nil
	}
	game, found := s.coord.Games().Get(turn.GameID)
	if !found {
		s.coord.Leases().InvalidateTurn(lease.TurnID)
		return nil, false, nil
	}
	return &Claim{
		ClaimID:   lease.ClaimID,
		TurnID:    lease.TurnID,
		ExpiresAt: lease.ExpiresAt.UTC(),
		GameID:    game.ID,
		Version:   game.Version,
		Player:    int(game.Current),
		Board:     game.Board.Cells(),
		History:   game.History(),
		LastMove:  game.LastMove,
		Config:    turn.Config,
	}, true, nil
}

func (s *ClaimService) wait(ctx context.Context, remaining time.Duration) error {
	s.coord.Lock()
	signal := s.coord.WaitChannel()
	s.coord.Unlock()

	timer := time.NewTimer(remaining)
	defer timer.Stop()
	select {
	case <-signal:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (s *ClaimService) Renew(claimID, engineID string, extendSeconds int) (*scheduler.Lease, error) {
	s.coord.Lock()
	defer s.coord.Unlock()
	lease, err := s.coord.Leases().Renew(claimID, engineID, extendSeconds)
	if err != nil {
		return nil, mapLeaseError(err)
	}
	return lease, nil
}

func (s *ClaimService) Release(claimID, engineID string) error {
	s.coord.Lock()
	defer s.coord.Unlock()
	if err := s.coord.Leases().Release(claimID, engineID); err != nil {
		return mapLeaseError(err)
	}
	s.coord.Signal()
	return nil
}

func mapLeaseError(err error) error {
	switch err {
	case scheduler.ErrClaimNotFound:
		return newError(CodeClaimNotFound, "任务租约不存在")
	case scheduler.ErrClaimExpired:
		return newError(CodeClaimExpired, "任务租约已过期")
	case scheduler.ErrClaimInvalid:
		return newError(CodeClaimInvalid, "任务租约无效")
	case scheduler.ErrForbidden:
		return newError(CodeForbidden, "无权操作该任务租约")
	case scheduler.ErrLeaseTooLong:
		return newError(CodeClaimInvalid, "任务租约总时长已达上限")
	default:
		return newErrorf(CodeInternalError, "租约操作失败: %v", err)
	}
}
