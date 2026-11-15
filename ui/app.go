package ui

import (
	"errors"
	"fmt"
	"sync"
	"time"

	mygo "github.com/egoist/mygo"
	mygoui "github.com/egoist/mygo/ui"

	"mygo/internal/application"
	"mygo/internal/auth"
	"mygo/internal/domain/gomoku"
)

// state is the interface state. It never holds the authoritative game state,
// only the snapshot the view draws and the tokens the interface owns.
type state struct {
	modeIndex   int
	engineIndex int
	aiIsBlack   bool
	gameID      string
	ownerToken  string
	game        *gomoku.Game
	history     []gomoku.Move
	status      string
	notice      string
	noticeKind  noticeKind
}

// noticeKind tells how a notice is shown.
type noticeKind int

const (
	noticeNone noticeKind = iota
	noticeInfo
	noticeError
)

// engineInfo describes one engine of the registry for the interface.
type engineInfo struct {
	ID      string
	Name    string
	Enabled bool
}

// App is the desktop interface of MyGo. It reads the game through the
// application services and never changes the authoritative state itself.
type App struct {
	games   *application.GameService
	moves   *application.MoveService
	engines []engineInfo

	mu    sync.Mutex
	st    state
	board *BoardView
	win   *mygo.Window
	// list keeps the place of the move list, so it follows the last move.
	list mygoui.ListState
}

// New builds the interface over the application services.
func New(games *application.GameService, moves *application.MoveService, registry *auth.Registry) *App {
	app := &App{games: games, moves: moves, board: newBoardView()}
	for _, engine := range registry.Engines() {
		app.engines = append(app.engines, engineInfo{ID: engine.EngineID, Name: engine.DisplayName, Enabled: engine.Available})
	}
	app.st.status = "请选择模式并开始一局新棋"
	return app
}

// View returns the view function of the window.
func (a *App) View() func(c *mygoui.Context) {
	return func(c *mygoui.Context) { a.build(c) }
}

// AttachWindow gives the interface the window it draws in, so it can ask for
// a repaint from a background goroutine.
func (a *App) AttachWindow(win *mygo.Window) {
	a.mu.Lock()
	a.win = win
	a.mu.Unlock()
}

// StartPolling refreshes the snapshot of the current game until the window
// is closed.
func (a *App) StartPolling() {
	go func() {
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for range ticker.C {
			a.mu.Lock()
			win := a.win
			id := a.st.gameID
			a.mu.Unlock()
			if win == nil {
				continue
			}
			if id == "" {
				continue
			}
			a.refresh()
			win.Invalidate()
		}
	}()
}

// Mode returns the selected game mode.
func (a *App) Mode() gomoku.Mode {
	a.mu.Lock()
	defer a.mu.Unlock()
	return modeValues[a.st.modeIndex]
}

// AIIsBlack reports whether the AI takes the black stones in a human game.
func (a *App) AIIsBlack() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.st.aiIsBlack
}

// EngineID returns the engine selected for the AI side.
func (a *App) EngineID() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.engines) == 0 {
		return ""
	}
	return a.engines[a.st.engineIndex%len(a.engines)].ID
}

// otherEngineID returns the engine that is not the given one, so an AI game
// uses both registered engines.
func (a *App) otherEngineID(current string) string {
	for _, engine := range a.engines {
		if engine.ID != current && engine.Enabled {
			return engine.ID
		}
	}
	return current
}

// newGame creates a game through the application service.
func (a *App) newGame() {
	mode := a.Mode()
	engine := a.EngineID()
	aiIsBlack := a.AIIsBlack()
	players := map[string]application.CreatePlayer{
		"black": {Type: gomoku.TypeHuman, Config: map[string]any{}},
		"white": {Type: gomoku.TypeHuman, Config: map[string]any{}},
	}
	switch mode {
	case gomoku.ModeHumanVsAI:
		aiSide := "white"
		if aiIsBlack {
			aiSide = "black"
		}
		players[aiSide] = application.CreatePlayer{Type: gomoku.TypeAI, EngineID: engine, Config: map[string]any{}}
	case gomoku.ModeAIVsAI:
		players["black"] = application.CreatePlayer{Type: gomoku.TypeAI, EngineID: engine, Config: map[string]any{}}
		players["white"] = application.CreatePlayer{Type: gomoku.TypeAI, EngineID: a.otherEngineID(engine), Config: map[string]any{}}
	}
	result, err := a.games.Create("", application.CreateGameRequest{Mode: mode, Players: players}, "")
	if err != nil {
		a.setError(err)
		return
	}
	a.mu.Lock()
	a.st.gameID = result.Game.ID
	a.st.ownerToken = result.OwnerToken
	a.st.game = result.Game
	a.st.history = nil
	a.st.notice = ""
	a.st.noticeKind = noticeNone
	a.st.status = describeGame(result.Game)
	a.mu.Unlock()
	a.refresh()
}

// pick applies a move of the human through the application service.
func (a *App) pick(row, col int) {
	a.mu.Lock()
	game := a.st.game
	owner := a.st.ownerToken
	a.mu.Unlock()
	if game == nil || owner == "" {
		return
	}
	if game.Status != gomoku.StatusPlaying {
		a.setNotice(noticeInfo, "本局已经结束，请开始新棋局")
		return
	}
	caller := application.Caller{Kind: application.CallerOwner, OwnerTokenHash: auth.HashToken(owner)}
	result, err := a.moves.Submit(game.ID, caller, application.SubmitMoveRequest{
		Row:             row,
		Col:             col,
		ExpectedVersion: game.Version,
	})
	if err != nil {
		// The board may be one poll behind the AI. Refresh once and retry.
		if appErr, ok := err.(*application.Error); ok && appErr.Code == application.CodeVersionConflict {
			a.refresh()
			if fresh := a.snapshot(); fresh != nil && fresh.Version != game.Version {
				result, err = a.moves.Submit(game.ID, caller, application.SubmitMoveRequest{
					Row:             row,
					Col:             col,
					ExpectedVersion: fresh.Version,
				})
			}
		}
		if err != nil {
			a.setError(err)
			return
		}
	}
	a.mu.Lock()
	a.st.game = result.Game
	a.st.status = describeGame(result.Game)
	a.st.notice = ""
	a.st.noticeKind = noticeNone
	a.mu.Unlock()
	a.refresh()
}

// togglePause pauses or resumes an AI game.
func (a *App) togglePause() {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()
	if game == nil || game.Mode != gomoku.ModeAIVsAI {
		return
	}
	state := gomoku.ExecutionRunning
	if game.Execution == gomoku.ExecutionRunning {
		state = gomoku.ExecutionPaused
	}
	updated, err := a.games.SetExecutionState(game.ID, state, nil)
	if err != nil {
		a.setError(err)
		return
	}
	a.mu.Lock()
	a.st.game = updated
	a.st.status = describeGame(updated)
	a.mu.Unlock()
}

// refresh reloads the snapshot and the history of the current game.
func (a *App) refresh() {
	a.mu.Lock()
	id := a.st.gameID
	a.mu.Unlock()
	if id == "" {
		return
	}
	game, err := a.games.Get(id)
	if err != nil {
		return
	}
	history, err := a.moves.History(id)
	if err != nil {
		return
	}
	a.mu.Lock()
	a.st.game = game
	a.st.history = history
	a.st.status = describeGame(game)
	a.mu.Unlock()
}

// snapshot returns the current game snapshot.
func (a *App) snapshot() *gomoku.Game {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.st.game
}

// setNotice shows a message in the interface.
func (a *App) setNotice(kind noticeKind, message string) {
	a.mu.Lock()
	a.st.noticeKind = kind
	a.st.notice = message
	a.mu.Unlock()
}

// setError shows an application error as a readable message.
func (a *App) setError(err error) {
	var appErr *application.Error
	if !errors.As(err, &appErr) {
		a.setError(err)
		return
	}
	a.setNotice(noticeError, errorMessage(appErr.Code))
}

// errorMessage returns the message of an application error code.
func errorMessage(code string) string {
	switch code {
	case application.CodeNotYourTurn:
		return "当前不是你的回合，等待对手落子"
	case application.CodePositionOccupied:
		return "该位置已经有棋子"
	case application.CodePositionOutOfRange:
		return "落点超出棋盘范围"
	case application.CodeVersionConflict:
		return "棋局已经变化，请重试"
	case application.CodeGameFinished:
		return "本局已经结束，请开始新棋局"
	case application.CodeGamePaused:
		return "棋局已暂停"
	case application.CodeForbidden:
		return "无权操作该棋局"
	case application.CodeClaimInvalid, application.CodeClaimExpired:
		return "AI 任务租约已失效"
	default:
		return code
	}
}

// playerLabel returns the Chinese name of a player color.
func playerLabel(player gomoku.Player) string {
	if player == gomoku.PlayerBlack {
		return "黑棋"
	}
	return "白棋"
}

// describeGame returns the status line of a game.
func describeGame(game *gomoku.Game) string {
	if game == nil {
		return ""
	}
	switch game.Status {
	case gomoku.StatusFinished:
		switch game.Result {
		case gomoku.ResultBlackWin:
			return "黑棋获胜"
		case gomoku.ResultWhiteWin:
			return "白棋获胜"
		case gomoku.ResultDraw:
			return "平局"
		}
		return "本局结束"
	}
	side := "黑棋"
	if game.Current == gomoku.PlayerWhite {
		side = "白棋"
	}
	who := "玩家"
	if sideConfig := game.ConfigFor(game.Current); sideConfig.IsAI() {
		who = "AI " + sideConfig.EngineName()
	}
	if game.Execution == gomoku.ExecutionPaused {
		return fmt.Sprintf("已暂停 · 轮到%s（%s）", side, who)
	}
	return fmt.Sprintf("第 %d 手 · 轮到%s（%s）", game.MoveCount+1, side, who)
}
