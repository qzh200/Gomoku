package ui

import (
	"fmt"

	mygo "github.com/egoist/mygo/ui"
	"github.com/lobehub/lobe-mygo/pkg/action"
	"github.com/lobehub/lobe-mygo/pkg/display"
	"github.com/lobehub/lobe-mygo/pkg/feedback"
	"github.com/lobehub/lobe-mygo/pkg/layout"
	"github.com/lobehub/lobe-mygo/pkg/lucide"
	"github.com/lobehub/lobe-mygo/pkg/theme"

	"mygo/internal/domain/gomoku"
)

// Panel and layout metrics.
const (
	sidebarWidth  = 284.0
	historyWidth  = 248.0
	pagePadding   = 20.0
	columnGap     = 16.0
	toolbarHeight = 56.0
	cardPadding   = 16.0
)

// modeNames are the labels of the game modes.
var modeNames = []string{"人机对战", "AI 对战", "人人对战"}

// modeValues are the game modes in the order of modeNames.
var modeValues = []gomoku.Mode{gomoku.ModeHumanVsAI, gomoku.ModeAIVsAI, gomoku.ModeHumanVsHuman}

// build draws the whole window.
func (a *App) build(c *mygo.Context) {
	theme.Apply(c)
	tk := theme.TokensOf(c)
	a.board.applyTheme(tk)

	page := mygo.Column(c).Fill()
	if !c.Vibrancy() {
		page.Background(tk.BgLayout)
	}
	page.Children(func() {
		a.buildToolbar(c, tk)
		mygo.Row(c).Grow(1).Basis(0).MinHeight(0).Padding(0, pagePadding, pagePadding, pagePadding).
			Gap(columnGap).AlignItems(mygo.Stretch).Children(func() {
			a.buildSidebar(c, tk)
			a.buildBoardArea(c, tk)
			a.buildHistory(c, tk)
		})
		feedback.ToastHost(c, feedback.ToastHostProps{})
	})
}

// buildToolbar draws the header of the window.
func (a *App) buildToolbar(c *mygo.Context, tk *theme.Tokens) {
	mygo.Row(c).Height(toolbarHeight).Padding(0, pagePadding).Gap(10).AlignItems(mygo.Center).Children(func() {
		a.buildToolbarTitle(c)
		mygo.Spacer(c).Grow(1).Basis(0)
		a.buildThinking(c)
		a.buildStatus(c, tk)
	})
}

// buildToolbarTitle shows the name of the app and the game it shows.
func (a *App) buildToolbarTitle(c *mygo.Context) {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()
	display.Icon(c, lucide.Grid3x3(), display.IconProps{Size: display.IconMiddle})
	display.Text(c, "MyGo 五子棋", display.TextProps{Heading: 4})
	if game == nil {
		return
	}
	display.Text(c, game.ID, display.TextProps{Type: display.TextSecondary, FontSize: 13})
}
func (a *App) buildThinking(c *mygo.Context) {
	a.mu.Lock()
	thinking := a.st.game != nil && a.st.game.AIThinking
	a.mu.Unlock()
	if !thinking {
		return
	}
	mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
		display.Icon(c, lucide.LoaderCircle(), display.IconProps{Size: display.IconSmall, Spin: true})
		display.Text(c, "AI 思考中", display.TextProps{Type: display.TextSecondary, FontSize: 13})
	})
}

// buildStatus shows the state of the current game.
func (a *App) buildStatus(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	game := a.st.game
	status := a.st.status
	a.mu.Unlock()
	if game == nil {
		display.Text(c, "选择模式并开始一局新棋", display.TextProps{Type: display.TextSecondary, FontSize: 13})
		return
	}
	color := display.TagBlue
	switch game.Status {
	case gomoku.StatusFinished:
		color = display.TagGreen
	}
	display.Tag(c, status, display.TagProps{Color: color, Round: true})
}

// buildSidebar holds the game controls and the state of the game.
func (a *App) buildSidebar(c *mygo.Context, tk *theme.Tokens) {
	mygo.Column(c).Width(sidebarWidth).AlignSelf(mygo.Start).Gap(columnGap).Children(func() {
		panel := layout.Block(c, layout.BlockProps{Variant: layout.BlockFilled}).Padding(16).Radius(tk.BorderRadiusLG)
		panel.Children(func() {
			mygo.Column(c).Gap(14).Children(func() {
				display.Text(c, "新对局", display.TextProps{Strong: true})
				a.buildModePicker(c)
				a.buildEnginePicker(c)
				a.buildNewGameButton(c)
				a.buildPauseButton(c)
				a.buildResult(c)
				a.buildNotice(c)
			})
		})
		a.buildGameInfo(c, tk)
	})
}

// buildModePicker lets the user pick the game mode.
func (a *App) buildModePicker(c *mygo.Context) {
	a.mu.Lock()
	index := a.st.modeIndex
	a.mu.Unlock()
	display.Text(c, "对战模式", display.TextProps{Type: display.TextSecondary, FontSize: 12})
	action.Segmented(c, &index, modeNames, action.SegmentedProps{Block: true})
	a.mu.Lock()
	a.st.modeIndex = index
	a.mu.Unlock()
}

// buildEnginePicker lets the user pick the engine of the AI side.
func (a *App) buildEnginePicker(c *mygo.Context) {
	a.mu.Lock()
	index := a.st.engineIndex
	mode := modeValues[a.st.modeIndex]
	a.mu.Unlock()
	if mode == gomoku.ModeHumanVsHuman {
		return
	}
	a.buildAISidePicker(c, mode)
	display.Text(c, "AI 引擎", display.TextProps{Type: display.TextSecondary, FontSize: 12})
	if len(a.engines) == 0 {
		display.Text(c, "没有可用引擎", display.TextProps{Type: display.TextDanger})
		return
	}
	names := make([]string, 0, len(a.engines))
	for _, engine := range a.engines {
		names = append(names, engine.Name)
	}
	action.Segmented(c, &index, names, action.SegmentedProps{Block: true})
	a.mu.Lock()
	a.st.engineIndex = index % len(a.engines)
	a.mu.Unlock()
}

// buildAISidePicker lets the user pick the side the AI plays in a human game.
func (a *App) buildAISidePicker(c *mygo.Context, mode gomoku.Mode) {
	if mode != gomoku.ModeHumanVsAI {
		return
	}
	a.mu.Lock()
	black := a.st.aiIsBlack
	a.mu.Unlock()
	display.Text(c, "AI 执子", display.TextProps{Type: display.TextSecondary, FontSize: 12})
	sides := []string{"黑方", "白方"}
	index := 1
	if black {
		index = 0
	}
	action.Segmented(c, &index, sides, action.SegmentedProps{Block: true})
	a.mu.Lock()
	a.st.aiIsBlack = index == 0
	a.mu.Unlock()
}

// buildNewGameButton starts a new game.
func (a *App) buildNewGameButton(c *mygo.Context) {
	button := iconButton(c, lucide.Plus(), "开始新棋局", action.ButtonProps{Variant: action.Primary})
	if button.Clicked() {
		a.newGame()
	}
}

// buildPauseButton pauses or resumes an AI game.
func (a *App) buildPauseButton(c *mygo.Context) {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()
	if game == nil || game.Mode != gomoku.ModeAIVsAI {
		return
	}
	label := "暂停对战"
	icon := lucide.Pause()
	if game.Execution == gomoku.ExecutionPaused {
		label = "恢复对战"
		icon = lucide.Play()
	}
	button := iconButton(c, icon, label, action.ButtonProps{Variant: action.Default})
	if button.Clicked() {
		a.togglePause()
	}
}

// buildResult shows the outcome of a finished game.
func (a *App) buildResult(c *mygo.Context) {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()
	if game == nil || game.Status != gomoku.StatusFinished {
		return
	}
	alertType := feedback.Success
	title := "本局结束"
	switch game.Result {
	case gomoku.ResultBlackWin:
		title = "黑棋获胜"
	case gomoku.ResultWhiteWin:
		title = "白棋获胜"
	case gomoku.ResultDraw:
		title = "平局"
		alertType = feedback.Warning
	}
	feedback.Alert(c, title, feedback.AlertProps{Type: alertType, Variant: feedback.Soft})
}

// iconButton builds a button with an icon before its label. The button is
// built without a label of its own, so the label appears once.
func iconButton(c *mygo.Context, icon *mygo.SVG, label string, props action.ButtonProps) mygo.Element {
	button := action.StyleButton(c, mygo.ButtonBase(c), "", props)
	button.Children(func() {
		mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
			display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
			// The label takes the color of the button, not the one of the text.
			mygo.Text(c, label)
		})
	})
	return button
}

// buildNotice shows the last message of the interface.
func (a *App) buildNotice(c *mygo.Context) {
	a.mu.Lock()
	kind := a.st.noticeKind
	notice := a.st.notice
	a.mu.Unlock()
	if kind == noticeNone || notice == "" {
		return
	}
	alertType := feedback.Info
	if kind == noticeError {
		alertType = feedback.Error
	}
	feedback.Alert(c, notice, feedback.AlertProps{Type: alertType, Variant: feedback.Soft})
}

// buildGameInfo shows the mode, the players and the state of the game.
func (a *App) buildGameInfo(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()

	panel := layout.Block(c, layout.BlockProps{Variant: layout.BlockFilled}).Padding(16).Radius(tk.BorderRadiusLG)
	panel.Children(func() {
		mygo.Column(c).Gap(10).Children(func() {
			display.Text(c, "对局信息", display.TextProps{Strong: true})
			if game == nil {
				display.Text(c, "还没有进行中的棋局", display.TextProps{Type: display.TextSecondary, FontSize: 13})
				return
			}
			a.buildInfoRow(c, lucide.Target(), "模式", modeLabel(game.Mode))
			a.buildInfoRow(c, lucide.Circle(), "黑棋", sideLabel(game.Black()))
			a.buildInfoRow(c, lucide.Circle(), "白棋", sideLabel(game.White()))
			a.buildInfoRow(c, lucide.Dices(), "手数", fmt.Sprintf("%d 手", game.MoveCount))
			a.buildInfoRow(c, lucide.Info(), "状态", executionLabel(game))
		})
	})
}

// buildInfoRow shows one labeled value.
func (a *App) buildInfoRow(c *mygo.Context, icon *mygo.SVG, label, value string) {
	mygo.Row(c).Gap(8).AlignItems(mygo.Center).Children(func() {
		display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
		display.Text(c, label, display.TextProps{Type: display.TextSecondary, FontSize: 13})
		mygo.Spacer(c).Grow(1).Basis(0)
		display.Text(c, value, display.TextProps{FontSize: 13})
	})
}

// buildBoardArea draws the board, sized to the space the window leaves.
func (a *App) buildBoardArea(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()
	a.board.Game = game
	a.board.OnPick = a.pick

	card := layout.Block(c, layout.BlockProps{Variant: layout.BlockFilled}).Grow(1).Basis(0).
		Padding(16).Radius(tk.BorderRadiusLG)
	card.Children(func() {
		mygo.Box(c).Grow(1).Basis(0).Center().Children(func() {
			a.board.View(c, a.boardSize(c))
		})
	})
}

// boardSize returns the size of the board for the current window. It is the
// space the board card leaves for the board, so the board never overflows it.
func (a *App) boardSize(c *mygo.Context) float32 {
	width, height := c.Size()
	return boardSizeFor(width, height)
}

// boardSizeFor returns the size of the board for a window of the given size.
func boardSizeFor(width, height float32) float32 {
	horizontal := width - sidebarWidth - historyWidth - 2*columnGap - 2*pagePadding - 2*cardPadding
	vertical := height - toolbarHeight - 2*pagePadding - 2*cardPadding
	size := horizontal
	if vertical > 0 && vertical < size {
		size = vertical
	}
	if size > boardMaxSize {
		size = boardMaxSize
	}
	if size < boardMinSize {
		// A narrow window shrinks the board instead of letting it overflow.
		size = boardMinSize
		if horizontal < boardMinSize {
			size = horizontal
		}
		if vertical > 0 && vertical < size {
			size = vertical
		}
	}
	return size
}

// buildHistory lists the moves of the current game. The list is virtualized,
// so a long game scrolls instead of overflowing the window, and it follows
// the last move as the game goes on.
func (a *App) buildHistory(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	history := a.st.history
	game := a.st.game
	a.mu.Unlock()

	panel := layout.Block(c, layout.BlockProps{Variant: layout.BlockFilled}).Width(historyWidth).
		Padding(16).Radius(tk.BorderRadiusLG)
	panel.Children(func() {
		mygo.Column(c).Gap(10).Grow(1).Basis(0).Children(func() {
			display.Text(c, "落子记录", display.TextProps{Strong: true})
			if len(history) == 0 {
				display.Text(c, "还没有落子", display.TextProps{Type: display.TextSecondary, FontSize: 13})
				return
			}
			a.list.FollowEnd = true
			last := game != nil && game.LastMove != nil
			mygo.List(c, &a.list, len(history), func(i int) {
				a.buildHistoryRow(c, history[i], last && history[i].Number == game.LastMove.Number, tk)
			}).Grow(1).Basis(0)
		})
	})
}

// buildHistoryRow shows one move: its number, the color of its stone and
// where it was played.
func (a *App) buildHistoryRow(c *mygo.Context, move gomoku.Move, last bool, tk *theme.Tokens) {
	row := mygo.Row(c).Gap(8).AlignItems(mygo.Center).Padding(3, 6).Radius(6).Children(func() {
		a.buildStoneDot(c, move.Player, tk)
		display.Text(c, fmt.Sprintf("%d", move.Number), display.TextProps{Type: display.TextSecondary, FontSize: 12})
		display.Text(c, fmt.Sprintf("(%d, %d)", move.Row, move.Col), display.TextProps{FontSize: 13})
		mygo.Spacer(c).Grow(1).Basis(0)
		if last {
			display.Text(c, "最新", display.TextProps{Type: display.TextSecondary, FontSize: 11})
		}
	})
	if last {
		row.Background(tk.FillSecondary)
	}
}

// buildStoneDot draws the stone of a move. The white stone has a border, so
// it does not vanish into the panel.
func (a *App) buildStoneDot(c *mygo.Context, player gomoku.Player, tk *theme.Tokens) {
	dot := mygo.Box(c).Size(12, 12).Radius(6).Background(stoneColor(player)).Shrink(0)
	if player == gomoku.PlayerWhite {
		dot.Border(1, tk.Border)
	}
}

// modeLabel returns the name of a game mode.
func modeLabel(mode gomoku.Mode) string {
	switch mode {
	case gomoku.ModeAIVsAI:
		return "AI 对战"
	case gomoku.ModeHumanVsAI:
		return "人机对战"
	case gomoku.ModeHumanVsHuman:
		return "人人对战"
	}
	return string(mode)
}

// sideLabel returns the name of a side.
func sideLabel(config gomoku.PlayerConfig) string {
	if config.IsAI() {
		return "AI · " + config.EngineName()
	}
	return "玩家"
}

// executionLabel returns the state of the game.
func executionLabel(game *gomoku.Game) string {
	if game.Status == gomoku.StatusFinished {
		return "已结束"
	}
	if game.Execution == gomoku.ExecutionPaused {
		return "已暂停"
	}
	return "进行中"
}
