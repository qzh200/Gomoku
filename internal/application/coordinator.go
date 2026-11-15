package application

import (
	"sync"

	"mygo/internal/repository"
	"mygo/internal/scheduler"
)

// Coordinator owns the mutable state that several use cases share. It makes
// game updates and turn claims atomic while long polling waits happen
// outside of its lock.
type Coordinator struct {
	mu     sync.Mutex
	games  repository.GameRepository
	leases *scheduler.LeaseManager
	idem   repository.IdempotencyStore
	notify chan struct{}
}

// NewCoordinator builds a coordinator over the given storage.
func NewCoordinator(games repository.GameRepository, leases *scheduler.LeaseManager, idem repository.IdempotencyStore) *Coordinator {
	return &Coordinator{
		games:  games,
		leases: leases,
		idem:   idem,
		notify: make(chan struct{}),
	}
}

// Games returns the game repository.
func (c *Coordinator) Games() repository.GameRepository { return c.games }

// Leases returns the lease manager.
func (c *Coordinator) Leases() *scheduler.LeaseManager { return c.leases }

// Idempotency returns the idempotency store.
func (c *Coordinator) Idempotency() repository.IdempotencyStore { return c.idem }

// Lock takes the coordinator lock.
func (c *Coordinator) Lock() { c.mu.Lock() }

// Unlock releases the coordinator lock.
func (c *Coordinator) Unlock() { c.mu.Unlock() }

// Signal wakes every waiting claim poll. It must be called with the lock
// held.
func (c *Coordinator) Signal() {
	close(c.notify)
	c.notify = make(chan struct{})
}

// WaitChannel returns the channel that is closed on the next signal. It must
// be called with the lock held.
func (c *Coordinator) WaitChannel() <-chan struct{} { return c.notify }
