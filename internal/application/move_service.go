package application

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"mygo/internal/domain/gomoku"
	"mygo/internal/repository"
	"mygo/internal/scheduler"
)

// now returns the current time. It is a variable so tests can freeze it.
var now = time.Now

// CallerKind tells whether a caller acts as the owner of a game or as an AI
// engine.
type CallerKind int

// The caller kinds.
const (
	CallerOwner CallerKind = iota
	CallerEngine
)

// Caller identifies who submits a move.
type Caller struct {
	Kind CallerKind
	// EngineID is the engine of an AI caller, derived from its credential.
	EngineID string
	// OwnerTokenHash is the stored hash of the owner token of a local caller.
	OwnerTokenHash string
}

// SubmitMoveRequest is the request of the unified move use case.
type SubmitMoveRequest struct {
	Row             int
	Col             int
	ExpectedVersion int
	ClaimID         string
	// IdempotencyKey deduplicates retries of the same request.
	IdempotencyKey string
}

// MoveResult is the outcome of a successful move.
type MoveResult struct {
	Move gomoku.Move  `json:"move"`
	Game *gomoku.Game `json:"game"`
}

// MoveService owns the unified move use case. Human players and AI players
// share the same rules; only the identity check differs.
type MoveService struct {
	coord *Coordinator
}

// NewMoveService builds a move service.
func NewMoveService(coord *Coordinator) *MoveService {
	return &MoveService{coord: coord}
}

// Submit validates and applies a move. A repeated request with the same
// idempotency key returns the saved result without placing a second stone.
func (s *MoveService) Submit(gameID string, caller Caller, req SubmitMoveRequest) (*MoveResult, error) {
	scope := IdempotencyScope(caller, "POST /games/"+gameID+"/moves", req.IdempotencyKey)
	hash := RequestHash(req.Row, req.Col, req.ExpectedVersion, req.ClaimID)

	s.coord.Lock()
	defer s.coord.Unlock()

	if record, ok := s.coord.Idempotency().Get(scope); ok {
		if record.RequestHash != hash {
			return nil, newError(CodeIdempotencyReused, "幂等键已用于不同的请求")
		}
		result, ok := record.Response.(*MoveResult)
		if !ok {
			return nil, newError(CodeInternalError, "幂等记录损坏")
		}
		return result, nil
	}

	game, ok := s.coord.Games().Get(gameID)
	if !ok {
		return nil, newError(CodeGameNotFound, "棋局不存在")
	}
	if err := s.checkCaller(game, caller); err != nil {
		return nil, err
	}
	if err := s.checkTurn(game, caller); err != nil {
		return nil, err
	}
	if game.Status == gomoku.StatusFinished {
		return nil, newError(CodeGameFinished, "棋局已经结束")
	}
	if game.Execution == gomoku.ExecutionPaused {
		return nil, newError(CodeGamePaused, "棋局已经暂停")
	}
	if !gomoku.InBounds(req.Row, req.Col) {
		return nil, newError(CodePositionOutOfRange, "落点超出棋盘范围")
	}
	if !game.Board.IsEmpty(req.Row, req.Col) {
		return nil, newError(CodePositionOccupied, "该位置已经有棋子")
	}
	if req.ExpectedVersion != game.Version {
		return nil, newError(CodeVersionConflict, "棋局版本已经变化")
	}
	if caller.Kind == CallerEngine {
		if err := s.checkClaim(game, caller, req.ClaimID); err != nil {
			return nil, err
		}
	}

	player := game.Current
	var result *MoveResult
	err := s.coord.Games().Update(gameID, func(stored *gomoku.Game) error {
		if stored.Version != game.Version {
			return newError(CodeVersionConflict, "棋局版本已经变化")
		}
		if !stored.Place(req.Row, req.Col, player) {
			return newError(CodePositionOccupied, "该位置已经有棋子")
		}
		move := stored.History()[stored.MoveCount-1]
		result = &MoveResult{Move: move, Game: stored.Clone()}
		return nil
	})
	if err != nil {
		if appErr, ok := err.(*Error); ok {
			return nil, appErr
		}
		if err == repository.ErrNotFound {
			return nil, newError(CodeGameNotFound, "棋局不存在")
		}
		return nil, newErrorf(CodeInternalError, "落子失败: %v", err)
	}

	result.Game.AIThinking = s.coord.Leases().ActiveFor(gameID, result.Game.Version)
	s.coord.Leases().InvalidateTurn(scheduler.TurnID(gameID, game.Version))
	s.coord.Signal()
	if req.IdempotencyKey != "" {
		s.coord.Idempotency().Put(scope, repository.IdempotencyRecord{RequestHash: hash, Response: result})
	}
	return result, nil
}

// checkCaller verifies that the caller may act on the game at all.
func (s *MoveService) checkCaller(game *gomoku.Game, caller Caller) error {
	switch caller.Kind {
	case CallerOwner:
		stored, ok := s.coord.Games().OwnerTokenHash(game.ID)
		if !ok || stored != caller.OwnerTokenHash {
			return newError(CodeForbidden, "无权操作该棋局")
		}
	case CallerEngine:
		if !s.playsEngine(game, caller.EngineID) {
			return newError(CodeForbidden, "该引擎不是本棋局的 AI")
		}
	default:
		return newError(CodeUnauthorized, "缺少有效凭据")
	}
	return nil
}

// playsEngine reports whether the engine plays one of the sides.
func (s *MoveService) playsEngine(game *gomoku.Game, engineID string) bool {
	for _, side := range []gomoku.PlayerConfig{game.Black(), game.White()} {
		if side.IsAI() && side.EngineName() == engineID {
			return true
		}
	}
	return false
}

// checkTurn verifies that the caller owns the side to move.
func (s *MoveService) checkTurn(game *gomoku.Game, caller Caller) error {
	side := game.ConfigFor(game.Current)
	switch caller.Kind {
	case CallerOwner:
		if side.IsAI() {
			return newError(CodeNotYourTurn, "当前不是人类的回合")
		}
	case CallerEngine:
		if !side.IsAI() || side.EngineName() != caller.EngineID {
			return newError(CodeNotYourTurn, "当前不是该引擎的回合")
		}
	}
	return nil
}

// checkClaim verifies that the AI holds a valid lease for the turn.
func (s *MoveService) checkClaim(game *gomoku.Game, caller Caller, claimID string) error {
	if claimID == "" {
		return newError(CodeClaimInvalid, "AI 落子必须携带 claim_id")
	}
	lease, ok := s.coord.Leases().Get(claimID)
	if !ok {
		return newError(CodeClaimInvalid, "任务租约不存在")
	}
	if lease.EngineID != caller.EngineID {
		return newError(CodeForbidden, "无权使用该任务租约")
	}
	if lease.GameID != game.ID {
		return newError(CodeClaimInvalid, "任务租约不属于该棋局")
	}
	if lease.Version != game.Version {
		return newError(CodeClaimInvalid, "任务租约已经过期")
	}
	if lease.Expired(now()) {
		return newError(CodeClaimExpired, "任务租约已过期")
	}
	return nil
}

// CheckOwner verifies that the caller owns the game.
func (s *MoveService) CheckOwner(gameID string, caller Caller) (*gomoku.Game, error) {
	s.coord.Lock()
	defer s.coord.Unlock()
	game, ok := s.coord.Games().Get(gameID)
	if !ok {
		return nil, newError(CodeGameNotFound, "棋局不存在")
	}
	if err := s.checkCaller(game, caller); err != nil {
		return nil, err
	}
	return game, nil
}

// History returns the moves of a game.
func (s *MoveService) History(gameID string) ([]gomoku.Move, error) {
	s.coord.Lock()
	defer s.coord.Unlock()
	game, ok := s.coord.Games().Get(gameID)
	if !ok {
		return nil, newError(CodeGameNotFound, "棋局不存在")
	}
	return game.History(), nil
}

// IdempotencyScope builds the storage scope of an idempotency key.
func IdempotencyScope(caller Caller, endpoint, key string) string {
	who := caller.OwnerTokenHash
	if caller.Kind == CallerEngine {
		who = caller.EngineID
	}
	return fmt.Sprintf("%s|%s|%s", who, endpoint, key)
}

// RequestHash identifies the content of a mutating request.
func RequestHash(parts ...any) string {
	sum := sha256.Sum256([]byte(fmt.Sprint(parts...)))
	return hex.EncodeToString(sum[:])
}
