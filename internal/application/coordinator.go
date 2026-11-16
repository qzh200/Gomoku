package application

import (
	"sync"

	"mygo/internal/domain/gomoku"
	"mygo/internal/repository"
	"mygo/internal/scheduler"
)

type Coordinator struct {
	mu     sync.Mutex
	games  repository.GameRepository
	leases *scheduler.LeaseManager
	idem   repository.IdempotencyStore
	notify chan struct{}
}

func NewCoordinator(games repository.GameRepository, leases *scheduler.LeaseManager, idem repository.IdempotencyStore) *Coordinator {
	return &Coordinator{
		games:  games,
		leases: leases,
		idem:   idem,
		notify: make(chan struct{}),
	}
}

func (c *Coordinator) Games() repository.GameRepository { return c.games }

func (c *Coordinator) Leases() *scheduler.LeaseManager { return c.leases }

func (c *Coordinator) Idempotency() repository.IdempotencyStore { return c.idem }

func (c *Coordinator) Lock() { c.mu.Lock() }

func (c *Coordinator) Unlock() { c.mu.Unlock() }

func (c *Coordinator) Signal() {
	close(c.notify)
	c.notify = make(chan struct{})
}

func (c *Coordinator) WaitChannel() <-chan struct{} { return c.notify }

func (c *Coordinator) pendingTurns(engineID string) []scheduler.Turn {
	var turns []scheduler.Turn
	for _, game := range c.games.List() {
		if game.Status != gomoku.StatusPlaying || game.Execution != gomoku.ExecutionRunning {
			continue
		}
		side := game.ConfigFor(game.Current)
		if !side.IsAI() || side.EngineName() != engineID {
			continue
		}
		turns = append(turns, scheduler.Turn{
			GameID:   game.ID,
			Version:  game.Version,
			Player:   int(game.Current),
			EngineID: engineID,
			Config:   side.Config,
		})
	}
	return turns
}
