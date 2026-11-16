package repository

import "mygo/internal/domain/gomoku"

type GameRepository interface {
	Create(game *gomoku.Game, ownerTokenHash string) error

	Get(id string) (*gomoku.Game, bool)

	List() []*gomoku.Game

	Delete(id string) bool

	Update(id string, fn func(*gomoku.Game) error) error

	OwnerTokenHash(id string) (string, bool)
}

type IdempotencyRecord struct {
	RequestHash string

	Response any
}

type IdempotencyStore interface {
	Get(scope string) (IdempotencyRecord, bool)

	Put(scope string, record IdempotencyRecord)
}
