package gomoku

// Mode is a game mode.
type Mode string

// The supported game modes.
const (
	ModeAIVsAI       Mode = "ai_vs_ai"
	ModeHumanVsAI    Mode = "human_vs_ai"
	ModeHumanVsHuman Mode = "human_vs_human"
)

// Valid reports whether the mode is one of the supported modes.
func (m Mode) Valid() bool {
	switch m {
	case ModeAIVsAI, ModeHumanVsAI, ModeHumanVsHuman:
		return true
	}
	return false
}

// PlayerType tells whether a side is played by a human or by an AI.
type PlayerType string

// The supported player types.
const (
	TypeHuman PlayerType = "human"
	TypeAI    PlayerType = "ai"
)

// Status is the lifecycle state of a game.
type Status string

// The game statuses.
const (
	StatusPlaying  Status = "playing"
	StatusFinished Status = "finished"
)

// Result is the outcome of a game.
type Result string

// The game results.
const (
	ResultOngoing  Result = "ongoing"
	ResultBlackWin Result = "black_win"
	ResultWhiteWin Result = "white_win"
	ResultDraw     Result = "draw"
)

// ExecutionState tells whether an automated game is running or paused.
type ExecutionState string

// The execution states.
const (
	ExecutionRunning ExecutionState = "running"
	ExecutionPaused  ExecutionState = "paused"
)

// PlayerConfig describes one side of a game.
//
// The JSON shape follows the frozen API contract: engine_id is null for human
// players and config is always an object, never null.
type PlayerConfig struct {
	Type     PlayerType     `json:"type"`
	EngineID *string        `json:"engine_id"`
	Config   map[string]any `json:"config"`
}

// IsAI reports whether the side is played by an AI.
func (p PlayerConfig) IsAI() bool { return p.Type == TypeAI }

// EngineName returns the engine identifier of an AI side, or "" for a human.
func (p PlayerConfig) EngineName() string {
	if p.EngineID == nil {
		return ""
	}
	return *p.EngineID
}

// Game is the authoritative state of one game.
//
// The JSON shape follows the frozen API contract, so the domain entity is the
// wire model: the board is a full 15x15 grid, the winner is null until a
// player wins and ai_thinking is computed by MyGo.
type Game struct {
	ID         string                  `json:"id"`
	Mode       Mode                    `json:"mode"`
	Players    map[string]PlayerConfig `json:"players"`
	Board      Board                   `json:"board"`
	Current    Player                  `json:"current_player"`
	Status     Status                  `json:"status"`
	Result     Result                  `json:"result"`
	Winner     *Player                 `json:"winner"`
	LastMove   *Move                   `json:"last_move"`
	MoveCount  int                     `json:"move_count"`
	Version    int                     `json:"version"`
	Execution  ExecutionState          `json:"execution_state"`
	AIThinking bool                    `json:"ai_thinking"`

	moves []Move
}

// Black returns the configuration of the black side.
func (g *Game) Black() PlayerConfig { return g.Players["black"] }

// White returns the configuration of the white side.
func (g *Game) White() PlayerConfig { return g.Players["white"] }

// ConfigFor returns the configuration of the side playing the given color.
func (g *Game) ConfigFor(p Player) PlayerConfig {
	if p == PlayerBlack {
		return g.Black()
	}
	return g.White()
}

// SideOf returns the side name of the given color.
func SideOf(p Player) string {
	if p == PlayerBlack {
		return "black"
	}
	return "white"
}

// NewGame builds a game with an empty board and the black side to move.
func NewGame(id string, mode Mode, black, white PlayerConfig) *Game {
	return &Game{
		ID:        id,
		Mode:      mode,
		Players:   map[string]PlayerConfig{"black": black, "white": white},
		Board:     NewBoard(),
		Current:   PlayerBlack,
		Status:    StatusPlaying,
		Result:    ResultOngoing,
		MoveCount: 0,
		Version:   0,
		Execution: ExecutionRunning,
	}
}

// Clone returns a copy of the game that shares no mutable state with it.
func (g *Game) Clone() *Game {
	clone := *g
	clone.Players = make(map[string]PlayerConfig, len(g.Players))
	for side, config := range g.Players {
		clone.Players[side] = config
	}
	clone.moves = g.History()
	if clone.LastMove != nil {
		last := *clone.LastMove
		clone.LastMove = &last
	}
	if clone.Winner != nil {
		winner := *clone.Winner
		clone.Winner = &winner
	}
	return &clone
}

// History returns the moves of the game in the order they were played.
func (g *Game) History() []Move {
	out := make([]Move, len(g.moves))
	copy(out, g.moves)
	return out
}

// Place writes a stone and updates the game state. It reports false when the
// position is not free or out of bounds.
func (g *Game) Place(row, col int, player Player) bool {
	if !g.Board.Place(row, col, player) {
		return false
	}
	g.moves = append(g.moves, Move{Row: row, Col: col, Player: player, Number: g.MoveCount + 1})
	g.MoveCount++
	g.Version++
	g.LastMove = &g.moves[len(g.moves)-1]
	g.Current = player.Other()

	if WinsAt(g.Board, row, col, player) {
		winner := player
		g.Status = StatusFinished
		g.Winner = &winner
		if player == PlayerBlack {
			g.Result = ResultBlackWin
		} else {
			g.Result = ResultWhiteWin
		}
		return true
	}
	g.resolveDraw()
	return true
}

// resolveDraw ends the game as a draw when the board is full.
func (g *Game) resolveDraw() {
	if g.Status == StatusFinished || !g.Board.IsFull() {
		return
	}
	g.Status = StatusFinished
	g.Result = ResultDraw
}
