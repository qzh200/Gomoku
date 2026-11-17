package gomoku

type Mode string

const (
	ModeAIVsAI       Mode = "ai_vs_ai"
	ModeHumanVsAI    Mode = "human_vs_ai"
	ModeHumanVsHuman Mode = "human_vs_human"
)

func (m Mode) Valid() bool {
	switch m {
	case ModeAIVsAI, ModeHumanVsAI, ModeHumanVsHuman:
		return true
	}
	return false
}

type PlayerType string

const (
	TypeHuman PlayerType = "human"
	TypeAI    PlayerType = "ai"
)

type Status string

const (
	StatusPlaying  Status = "playing"
	StatusFinished Status = "finished"
)

type Result string

const (
	ResultOngoing  Result = "ongoing"
	ResultBlackWin Result = "black_win"
	ResultWhiteWin Result = "white_win"
	ResultDraw     Result = "draw"
)

type ExecutionState string

const (
	ExecutionRunning ExecutionState = "running"
	ExecutionPaused  ExecutionState = "paused"
)

type PlayerConfig struct {
	Type     PlayerType     `json:"type"`
	EngineID *string        `json:"engine_id"`
	Config   map[string]any `json:"config"`
}

func (p PlayerConfig) IsAI() bool { return p.Type == TypeAI }

func (p PlayerConfig) EngineName() string {
	if p.EngineID == nil {
		return ""
	}
	return *p.EngineID
}

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

func (g *Game) Black() PlayerConfig { return g.Players["black"] }

func (g *Game) White() PlayerConfig { return g.Players["white"] }

func (g *Game) ConfigFor(p Player) PlayerConfig {
	if p == PlayerBlack {
		return g.Black()
	}
	return g.White()
}

func SideOf(p Player) string {
	if p == PlayerBlack {
		return "black"
	}
	return "white"
}

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

func (g *Game) History() []Move {
	out := make([]Move, len(g.moves))
	copy(out, g.moves)
	return out
}

func (g *Game) Undo() bool {
	if len(g.moves) == 0 {
		return false
	}
	version := g.Version + 1
	execution := g.Execution
	moves := g.moves[:len(g.moves)-1]
	*g = *NewGame(g.ID, g.Mode, g.Black(), g.White())
	for _, move := range moves {
		g.Place(move.Row, move.Col, move.Player)
	}
	g.Version = version
	g.Execution = execution
	return true
}

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

func (g *Game) resolveDraw() {
	if g.Status == StatusFinished || !g.Board.IsFull() {
		return
	}
	g.Status = StatusFinished
	g.Result = ResultDraw
}
