package scheduler

import (
	"time"
)

type Turn struct {
	GameID string

	Version int

	Player int

	EngineID string

	Config map[string]any
}

type TurnSource func(engineID string) []Turn

type Queue struct {
	leases *LeaseManager
	turns  TurnSource
	ttl    time.Duration
}

func NewQueue(leases *LeaseManager, turns TurnSource, ttl time.Duration) *Queue {
	if ttl <= 0 {
		ttl = DefaultLeaseSeconds * time.Second
	}
	return &Queue{leases: leases, turns: turns, ttl: ttl}
}

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
