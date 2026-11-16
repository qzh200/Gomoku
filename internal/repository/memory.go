package repository

import (
	"errors"
	"sync"

	"mygo/internal/domain/gomoku"
)

var (
	ErrAlreadyExists = errors.New("game already exists")
	ErrNotFound      = errors.New("game not found")
)

type memoryGames struct {
	mu     sync.RWMutex
	games  map[string]*gomoku.Game
	owners map[string]string
	order  []string
}

func NewMemoryGames() GameRepository {
	return &memoryGames{
		games:  make(map[string]*gomoku.Game),
		owners: make(map[string]string),
	}
}

func (m *memoryGames) Create(game *gomoku.Game, ownerTokenHash string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.games[game.ID]; ok {
		return ErrAlreadyExists
	}
	m.games[game.ID] = game
	m.owners[game.ID] = ownerTokenHash
	m.order = append(m.order, game.ID)
	return nil
}

func (m *memoryGames) Get(id string) (*gomoku.Game, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	game, ok := m.games[id]
	if !ok {
		return nil, false
	}
	return game.Clone(), true
}

func (m *memoryGames) List() []*gomoku.Game {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]*gomoku.Game, 0, len(m.order))
	for _, id := range m.order {
		if game, ok := m.games[id]; ok {
			out = append(out, game.Clone())
		}
	}
	return out
}

func (m *memoryGames) Delete(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.games[id]; !ok {
		return false
	}
	delete(m.games, id)
	delete(m.owners, id)
	for i, existing := range m.order {
		if existing == id {
			m.order = append(m.order[:i], m.order[i+1:]...)
			break
		}
	}
	return true
}

func (m *memoryGames) Update(id string, fn func(*gomoku.Game) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	game, ok := m.games[id]
	if !ok {
		return ErrNotFound
	}
	return fn(game)
}

func (m *memoryGames) OwnerTokenHash(id string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	hash, ok := m.owners[id]
	return hash, ok
}

type memoryIdempotency struct {
	mu      sync.Mutex
	records map[string]IdempotencyRecord
}

func NewMemoryIdempotency() IdempotencyStore {
	return &memoryIdempotency{records: make(map[string]IdempotencyRecord)}
}

func (m *memoryIdempotency) Get(scope string) (IdempotencyRecord, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	record, ok := m.records[scope]
	return record, ok
}

func (m *memoryIdempotency) Put(scope string, record IdempotencyRecord) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.records[scope] = record
}
