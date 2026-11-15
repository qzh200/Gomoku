// Package repository abstracts the storage of games and idempotency records.
package repository

import "mygo/internal/domain/gomoku"

// GameRepository stores the authoritative game snapshots.
type GameRepository interface {
	// Create stores a new game. It returns ErrAlreadyExists when the id is taken.
	Create(game *gomoku.Game, ownerTokenHash string) error
	// Get returns a snapshot of the game, or false when it does not exist.
	Get(id string) (*gomoku.Game, bool)
	// List returns snapshots of all games, in creation order.
	List() []*gomoku.Game
	// Delete removes a game. It reports whether the game existed.
	Delete(id string) bool
	// Update applies fn to the stored game under the repository lock. It
	// returns ErrNotFound when the game does not exist and the error of fn
	// when fn fails.
	Update(id string, fn func(*gomoku.Game) error) error
	// OwnerTokenHash returns the stored hash of the owner token of a game.
	OwnerTokenHash(id string) (string, bool)
}

// IdempotencyRecord is a saved response of a mutating request.
type IdempotencyRecord struct {
	// RequestHash identifies the request body the response belongs to.
	RequestHash string
	// Response is the value returned to a retry of the same request.
	Response any
}

// IdempotencyStore saves responses of mutating requests, scoped by caller
// and endpoint.
type IdempotencyStore interface {
	// Get returns the record stored under scope.
	Get(scope string) (IdempotencyRecord, bool)
	// Put stores a record under scope, replacing an older one.
	Put(scope string, record IdempotencyRecord)
}
