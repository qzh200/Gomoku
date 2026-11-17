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

type state struct {
	modeIndex   int
	engineIndex int
	aiIsBlack   bool

	firstEngineIndex  int
	secondEngineIndex int

	gameID     string
	ownerToken string
	game       *gomoku.Game
	history    []gomoku.Move
	status     string
	notice     string
	noticeKind noticeKind

	replayGames []*gomoku.Game
	reviewing   bool
	reviewGame  *gomoku.Game
	reviewMoves []gomoku.Move
	reviewStep  int
}

type noticeKind int

const (
	noticeNone noticeKind = iota
	noticeInfo
	noticeError
)

type engineInfo struct {
	ID      string
	Name    string
	Enabled bool
}

type App struct {
	games   *application.GameService
	moves   *application.MoveService
	engines []engineInfo

	mu    sync.Mutex
	st    state
	board *BoardView
	win   *mygo.Window

	list       mygoui.ListState
	replayList mygoui.ListState
	reviewList mygoui.ListState
}

func New(games *application.GameService, moves *application.MoveService, registry *auth.Registry) *App {
	app := &App{games: games, moves: moves, board: newBoardView()}
	for _, engine := range registry.Engines() {
		app.engines = append(app.engines, engineInfo{ID: engine.EngineID, Name: engine.DisplayName, Enabled: engine.Available})
	}
	app.st.status = "请选择模式并开始一局新棋"
	if len(app.engines) > 1 {

		app.st.secondEngineIndex = 1
	}
	return app
}

func (a *App) View() func(c *mygoui.Context) {
	return func(c *mygoui.Context) { a.build(c) }
}

func (a *App) AttachWindow(win *mygo.Window) {
	a.mu.Lock()
	a.win = win
	a.mu.Unlock()
}

func (a *App) StartPolling() {
	go func() {
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for range ticker.C {
			a.mu.Lock()
			win := a.win
			a.mu.Unlock()
			if win == nil {
				continue
			}
			a.refresh()
			win.Invalidate()
		}
	}()
}

func (a *App) Mode() gomoku.Mode {
	a.mu.Lock()
	defer a.mu.Unlock()
	return modeValues[a.st.modeIndex]
}

func (a *App) AIIsBlack() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.st.aiIsBlack
}

func (a *App) EngineID() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.engines) == 0 {
		return ""
	}
	return a.engines[a.st.engineIndex%len(a.engines)].ID
}

func (a *App) FirstEngineID() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.engines) == 0 {
		return ""
	}
	return a.engines[a.st.firstEngineIndex%len(a.engines)].ID
}

func (a *App) SecondEngineID() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.engines) == 0 {
		return ""
	}
	return a.engines[a.st.secondEngineIndex%len(a.engines)].ID
}

func (a *App) newGame() {
	a.mu.Lock()
	previous := a.st.game
	a.mu.Unlock()
	if previous != nil && previous.Status == gomoku.StatusPlaying && previous.MoveCount == 0 {
		_ = a.games.Delete(previous.ID)
	}
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
		players["black"] = application.CreatePlayer{Type: gomoku.TypeAI, EngineID: a.FirstEngineID(), Config: map[string]any{}}
		players["white"] = application.CreatePlayer{Type: gomoku.TypeAI, EngineID: a.SecondEngineID(), Config: map[string]any{}}
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
	a.stopReview()
	a.refresh()
}

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

func (a *App) undo() {
	a.mu.Lock()
	game := a.st.game
	owner := a.st.ownerToken
	a.mu.Unlock()
	if game == nil || owner == "" || game.Mode == gomoku.ModeAIVsAI {
		return
	}
	caller := application.Caller{Kind: application.CallerOwner, OwnerTokenHash: auth.HashToken(owner)}
	updated, err := a.moves.Undo(game.ID, caller, application.UndoRequest{ExpectedVersion: game.Version})
	if err != nil {
		if appErr, ok := err.(*application.Error); ok && appErr.Code == application.CodeVersionConflict {
			a.refresh()
			if fresh := a.snapshot(); fresh != nil && fresh.Version != game.Version {
				updated, err = a.moves.Undo(game.ID, caller, application.UndoRequest{ExpectedVersion: fresh.Version})
			}
		}
		if err != nil {
			a.setError(err)
			return
		}
	}
	a.mu.Lock()
	a.st.game = updated
	a.st.status = describeGame(updated)
	a.st.notice = ""
	a.st.noticeKind = noticeNone
	a.mu.Unlock()
	a.refresh()
}

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

func (a *App) refresh() {
	a.refreshReplay()
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

const replayLimit = 12

func (a *App) refreshReplay() {
	games := a.games.List()
	a.mu.Lock()
	currentID := a.st.gameID
	a.mu.Unlock()
	recent := make([]*gomoku.Game, 0, len(games))
	for i := len(games) - 1; i >= 0 && len(recent) < replayLimit; i-- {
		game := games[i]
		if game.Status != gomoku.StatusFinished && (game.ID == currentID || game.MoveCount == 0) {
			continue
		}
		recent = append(recent, game)
	}
	a.mu.Lock()
	a.st.replayGames = recent
	a.mu.Unlock()
}

func (a *App) startReview(game *gomoku.Game) {
	if game == nil {
		return
	}
	moves, err := a.moves.History(game.ID)
	if err != nil {
		a.setError(err)
		return
	}
	a.mu.Lock()
	a.st.reviewing = true
	a.st.reviewGame = game.Clone()
	a.st.reviewMoves = moves
	a.st.reviewStep = len(moves)
	a.mu.Unlock()
}

func (a *App) stopReview() {
	a.mu.Lock()
	a.st.reviewing = false
	a.st.reviewGame = nil
	a.st.reviewMoves = nil
	a.st.reviewStep = 0
	a.mu.Unlock()
}

func (a *App) stepReview(delta int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.setReviewStepLocked(a.st.reviewStep + delta)
}

func (a *App) setReviewStep(step int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.setReviewStepLocked(step)
}

func (a *App) setReviewStepLocked(step int) {
	if !a.st.reviewing {
		return
	}
	if step < 0 {
		step = 0
	}
	if step > len(a.st.reviewMoves) {
		step = len(a.st.reviewMoves)
	}
	a.st.reviewStep = step
}

func replayPosition(game *gomoku.Game, moves []gomoku.Move, step int) *gomoku.Game {
	if game == nil {
		return nil
	}
	if step < 0 {
		step = 0
	}
	if step > len(moves) {
		step = len(moves)
	}
	position := gomoku.NewGame(game.ID, game.Mode, game.Black(), game.White())
	for i := range step {
		position.Place(moves[i].Row, moves[i].Col, moves[i].Player)
	}
	position.AIThinking = false
	if step == len(moves) {
		position.Status = game.Status
		position.Result = game.Result
		position.Execution = game.Execution
		if game.Winner != nil {
			winner := *game.Winner
			position.Winner = &winner
		}
	}
	return position
}

func (a *App) snapshot() *gomoku.Game {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.st.game
}

func (a *App) setNotice(kind noticeKind, message string) {
	a.mu.Lock()
	a.st.noticeKind = kind
	a.st.notice = message
	a.mu.Unlock()
}

func (a *App) setError(err error) {
	var appErr *application.Error
	if !errors.As(err, &appErr) {
		a.setError(err)
		return
	}
	a.setNotice(noticeError, errorMessage(appErr.Code))
}

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
	case application.CodeInvalidRequest:
		return "无法悔棋"
	case application.CodeForbidden:
		return "无权操作该棋局"
	case application.CodeClaimInvalid, application.CodeClaimExpired:
		return "AI 任务租约已失效"
	default:
		return code
	}
}

func playerLabel(player gomoku.Player) string {
	if player == gomoku.PlayerBlack {
		return "黑棋"
	}
	return "白棋"
}

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
