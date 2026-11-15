package application

import (
	"fmt"
	"sync/atomic"

	"mygo/internal/auth"
	"mygo/internal/domain/gomoku"
	"mygo/internal/repository"
)

// CreatePlayer describes one side of a new game.
type CreatePlayer struct {
	Type     gomoku.PlayerType `json:"type"`
	EngineID string            `json:"engine_id"`
	Config   map[string]any    `json:"config"`
}

// CreateGameRequest is the request of the create game use case.
type CreateGameRequest struct {
	Mode    gomoku.Mode             `json:"mode"`
	Players map[string]CreatePlayer `json:"players"`
}

// CreateResult is the outcome of the create game use case.
type CreateResult struct {
	Game       *gomoku.Game `json:"game"`
	OwnerToken string       `json:"owner_token"`
}

// GameService owns the lifecycle of the games.
type GameService struct {
	coord   *Coordinator
	engines *auth.Registry
	nextID  atomic.Int64
}

// NewGameService builds a game service.
func NewGameService(coord *Coordinator, engines *auth.Registry) *GameService {
	return &GameService{coord: coord, engines: engines}
}

// Create creates a game and returns its snapshot with the owner token. The
// owner token is returned only here.
//
// The caller identity and the idempotency key scope the request, so a retry
// with the same key returns the same game instead of a second one.
func (s *GameService) Create(callerHash string, req CreateGameRequest, idempotencyKey string) (*CreateResult, error) {
	scope := IdempotencyScope(Caller{Kind: CallerOwner, OwnerTokenHash: callerHash}, "POST /games", idempotencyKey)
	hash := RequestHash(req.Mode, req.Players["black"], req.Players["white"])

	s.coord.Lock()
	defer s.coord.Unlock()

	if idempotencyKey != "" {
		if record, ok := s.coord.Idempotency().Get(scope); ok {
			if record.RequestHash != hash {
				return nil, newError(CodeIdempotencyReused, "幂等键已用于不同的请求")
			}
			result, ok := record.Response.(*CreateResult)
			if !ok {
				return nil, newError(CodeInternalError, "幂等记录损坏")
			}
			return result, nil
		}
	}

	result, err := s.create(req)
	if err != nil {
		return nil, err
	}
	if idempotencyKey != "" {
		s.coord.Idempotency().Put(scope, repository.IdempotencyRecord{RequestHash: hash, Response: result})
	}
	return result, nil
}

// create builds a new game. The caller must hold the coordinator lock.
func (s *GameService) create(req CreateGameRequest) (*CreateResult, error) {
	if !req.Mode.Valid() {
		return nil, newErrorf(CodeInvalidGameConfig, "未知的对战模式 %q", req.Mode)
	}
	black, err := s.playerConfig(req, "black")
	if err != nil {
		return nil, err
	}
	white, err := s.playerConfig(req, "white")
	if err != nil {
		return nil, err
	}
	if err := s.checkMode(req.Mode, black, white); err != nil {
		return nil, err
	}

	id := fmt.Sprintf("game_%03d", s.nextID.Add(1))
	game := gomoku.NewGame(id, req.Mode, black, white)
	ownerToken := auth.RandomToken()

	if err := s.coord.Games().Create(game, auth.HashToken(ownerToken)); err != nil {
		return nil, newErrorf(CodeInternalError, "创建棋局失败: %v", err)
	}
	return &CreateResult{Game: game.Clone(), OwnerToken: ownerToken}, nil
}

// playerConfig validates one side of a new game.
func (s *GameService) playerConfig(req CreateGameRequest, side string) (gomoku.PlayerConfig, error) {
	player, ok := req.Players[side]
	if !ok {
		return gomoku.PlayerConfig{}, newErrorf(CodeInvalidGameConfig, "缺少 %s 方玩家", side)
	}
	switch player.Type {
	case gomoku.TypeHuman:
		if player.EngineID != "" {
			return gomoku.PlayerConfig{}, newErrorf(CodeInvalidGameConfig, "%s 方为人类时不能指定引擎", side)
		}
		return gomoku.PlayerConfig{Type: gomoku.TypeHuman, Config: map[string]any{}}, nil
	case gomoku.TypeAI:
		if player.EngineID == "" {
			return gomoku.PlayerConfig{}, newErrorf(CodeInvalidGameConfig, "%s 方为 AI 时必须指定 engine_id", side)
		}
		engine, ok := s.engines.EngineByID(player.EngineID)
		if !ok {
			return gomoku.PlayerConfig{}, newErrorf(CodeInvalidGameConfig, "未注册的引擎 %q", player.EngineID)
		}
		if !engine.Available {
			return gomoku.PlayerConfig{}, newErrorf(CodeInvalidGameConfig, "引擎 %q 当前不可用", player.EngineID)
		}
		engineID := engine.EngineID
		config := player.Config
		if config == nil {
			config = map[string]any{}
		}
		return gomoku.PlayerConfig{Type: gomoku.TypeAI, EngineID: &engineID, Config: config}, nil
	default:
		return gomoku.PlayerConfig{}, newErrorf(CodeInvalidGameConfig, "%s 方类型无效", side)
	}
}

// checkMode verifies that the players match the mode.
func (s *GameService) checkMode(mode gomoku.Mode, black, white gomoku.PlayerConfig) error {
	human, ai := 0, 0
	for _, side := range []gomoku.PlayerConfig{black, white} {
		if side.IsAI() {
			ai++
		} else {
			human++
		}
	}
	switch mode {
	case gomoku.ModeAIVsAI:
		if ai != 2 {
			return newError(CodeInvalidGameConfig, "AI vs AI 模式要求双方都是 AI")
		}
	case gomoku.ModeHumanVsAI:
		if human != 1 || ai != 1 {
			return newError(CodeInvalidGameConfig, "Human vs AI 模式要求一方是人类、一方是 AI")
		}
	case gomoku.ModeHumanVsHuman:
		if human != 2 {
			return newError(CodeInvalidGameConfig, "Human vs Human 模式要求双方都是人类")
		}
	}
	return nil
}

// List returns the snapshots of all games.
func (s *GameService) List() []*gomoku.Game {
	s.coord.Lock()
	defer s.coord.Unlock()
	return s.withThinking(s.coord.Games().List())
}

// Get returns the snapshot of one game.
func (s *GameService) Get(id string) (*gomoku.Game, error) {
	s.coord.Lock()
	defer s.coord.Unlock()
	game, ok := s.coord.Games().Get(id)
	if !ok {
		return nil, newError(CodeGameNotFound, "棋局不存在")
	}
	game.AIThinking = s.coord.Leases().ActiveFor(game.ID, game.Version)
	return game, nil
}

// Delete removes a game and invalidates its leases.
func (s *GameService) Delete(id string) error {
	s.coord.Lock()
	defer s.coord.Unlock()
	if !s.coord.Games().Delete(id) {
		return newError(CodeGameNotFound, "棋局不存在")
	}
	s.coord.Leases().InvalidateGame(id)
	s.coord.Signal()
	return nil
}

// SetExecutionState pauses or resumes an automated game.
func (s *GameService) SetExecutionState(id string, state gomoku.ExecutionState, expectedVersion *int) (*gomoku.Game, error) {
	if state != gomoku.ExecutionPaused && state != gomoku.ExecutionRunning {
		return nil, newErrorf(CodeInvalidRequest, "未知的执行状态 %q", state)
	}
	s.coord.Lock()
	defer s.coord.Unlock()

	var failure error
	err := s.coord.Games().Update(id, func(game *gomoku.Game) error {
		if game.Mode != gomoku.ModeAIVsAI {
			failure = newError(CodeInvalidGameConfig, "只有 AI vs AI 模式支持暂停与恢复")
			return failure
		}
		if expectedVersion != nil && *expectedVersion != game.Version {
			failure = newError(CodeVersionConflict, "棋局版本已经变化")
			return failure
		}
		game.Execution = state
		return nil
	})
	if err != nil {
		if failure != nil {
			return nil, failure
		}
		if err == repository.ErrNotFound {
			return nil, newError(CodeGameNotFound, "棋局不存在")
		}
		return nil, newErrorf(CodeInternalError, "更新棋局失败: %v", err)
	}
	if state == gomoku.ExecutionPaused {
		s.coord.Leases().InvalidateGame(id)
	}
	s.coord.Signal()

	game, ok := s.coord.Games().Get(id)
	if !ok {
		return nil, newError(CodeGameNotFound, "棋局不存在")
	}
	game.AIThinking = s.coord.Leases().ActiveFor(game.ID, game.Version)
	return game, nil
}

// withThinking fills the ai_thinking flag of the snapshots.
func (s *GameService) withThinking(games []*gomoku.Game) []*gomoku.Game {
	for _, game := range games {
		game.AIThinking = s.coord.Leases().ActiveFor(game.ID, game.Version)
	}
	return games
}
