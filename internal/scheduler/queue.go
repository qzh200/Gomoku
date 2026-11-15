package scheduler

import (
	"time"
)

// Turn is a pending AI turn that a worker can take.
type Turn struct {
	// GameID is the game the turn belongs to.
	GameID string
	// Version is the version of the game at the turn.
	Version int
	// Player is the color the AI plays in this turn.
	Player int
	// EngineID is the engine the turn belongs to.
	EngineID string
	// Config is the private configuration of the engine for this game.
	Config map[string]any
}

// TurnSource lists the pending turns of the games. The application layer
// implements it over the game repository.
type TurnSource func(engineID string) []Turn

// Queue hands the pending turns of an engine to its workers. It owns the
// lease manager, so one turn is only ever leased once.
type Queue struct {
	leases *LeaseManager
	turns  TurnSource
	ttl    time.Duration
}

// NewQueue builds a queue over a lease manager and a turn source.
func NewQueue(leases *LeaseManager, turns TurnSource, ttl time.Duration) *Queue {
	if ttl <= 0 {
		ttl = DefaultLeaseSeconds * time.Second
	}
	return &Queue{leases: leases, turns: turns, ttl: ttl}
}

// Next leases the next pending turn of the engine. It reports false when the
// engine has no turn to take.
func (q *Queue) Next(engineID, workerID string) (*Lease, Turn, bool) {
	for _, turn := range q.turns(engineID) {
		turnID := TurnID(turn.GameID, turn.Version)
		if q.leases.ActiveFor(turn.GameID, turn.Version) {
			continue
		}
		lease, ok := q.leases.TryAcquire(turnID, turn.GameID, engineID, workerID, turn.Version, turn.Player, q.ttl)
		if !ok {
			continue
		}
		return lease, turn, true
	}
	return nil, Turn{}, false
}
