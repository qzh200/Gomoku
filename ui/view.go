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

const (
	sidebarWidth  = 284.0
	historyWidth  = 248.0
	pagePadding   = 20.0
	columnGap     = 16.0
	toolbarHeight = 56.0
	cardPadding   = 16.0
)

const (
	historyColumnGap   = 8.0
	historyNumberWidth = 32.0
	historyPlayerWidth = 60.0
	historyPlaceWidth  = 44.0
)

var modeNames = []string{"人机对战", "AI 对战", "人人对战"}

var modeValues = []gomoku.Mode{gomoku.ModeHumanVsAI, gomoku.ModeAIVsAI, gomoku.ModeHumanVsHuman}

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

func (a *App) buildToolbar(c *mygo.Context, tk *theme.Tokens) {
	mygo.Row(c).Height(toolbarHeight).Padding(0, pagePadding).Gap(10).AlignItems(mygo.Center).Children(func() {
		a.buildToolbarTitle(c)
		mygo.Spacer(c).Grow(1).Basis(0)
		a.buildThinking(c)
		a.buildStatus(c, tk)
	})
}

func (a *App) buildToolbarTitle(c *mygo.Context) {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()
	display.Icon(c, lucide.Grid3x3(), display.IconProps{Size: display.IconMiddle})
	display.Text(c, "五子棋", display.TextProps{Heading: 4})
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

func (a *App) buildEnginePicker(c *mygo.Context) {
	a.mu.Lock()
	mode := modeValues[a.st.modeIndex]
	a.mu.Unlock()
	switch mode {
	case gomoku.ModeHumanVsHuman:
		return
	case gomoku.ModeHumanVsAI:
		a.buildAISidePicker(c, mode)
		a.buildSingleEnginePicker(c, "AI 引擎", func() int {
			a.mu.Lock()
			defer a.mu.Unlock()
			return a.st.engineIndex
		}, func(index int) {
			a.mu.Lock()
			defer a.mu.Unlock()
			a.st.engineIndex = index
		})
	default:
		a.buildAIEnginePickers(c)
	}
}

func (a *App) buildSingleEnginePicker(c *mygo.Context, label string, current func() int, store func(int)) {
	display.Text(c, label, display.TextProps{Type: display.TextSecondary, FontSize: 12})
	names := a.engineNames()
	if len(names) == 0 {
		display.Text(c, "没有可用引擎", display.TextProps{Type: display.TextDanger})
		return
	}
	index := current() % len(names)
	action.Segmented(c, &index, names, action.SegmentedProps{Block: true})
	store(index % len(names))
}

func (a *App) buildAIEnginePickers(c *mygo.Context) {
	names := a.engineNames()
	if len(names) == 0 {
		display.Text(c, "没有可用引擎", display.TextProps{Type: display.TextDanger})
		return
	}
	a.mu.Lock()
	first := a.st.firstEngineIndex % len(names)
	second := a.st.secondEngineIndex % len(names)
	a.mu.Unlock()

	display.Text(c, "先手（黑方）", display.TextProps{Type: display.TextSecondary, FontSize: 12})
	action.Segmented(c, &first, names, action.SegmentedProps{Block: true})
	display.Text(c, "后手（白方）", display.TextProps{Type: display.TextSecondary, FontSize: 12})
	action.Segmented(c, &second, names, action.SegmentedProps{Block: true})

	if first == second && len(names) > 1 {
		second = (first + 1) % len(names)
	}
	a.mu.Lock()
	a.st.firstEngineIndex = first
	a.st.secondEngineIndex = second
	a.mu.Unlock()
}

func (a *App) engineNames() []string {
	names := make([]string, 0, len(a.engines))
	for _, engine := range a.engines {
		names = append(names, engine.Name)
	}
	return names
}

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

func (a *App) buildNewGameButton(c *mygo.Context) {
	button := iconButton(c, lucide.Plus(), "开始新棋局", action.ButtonProps{Variant: action.Primary})
	if button.Clicked() {
		a.newGame()
	}
}

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

func iconButton(c *mygo.Context, icon *mygo.SVG, label string, props action.ButtonProps) mygo.Element {
	button := action.StyleButton(c, mygo.ButtonBase(c), "", props)
	button.Children(func() {
		mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
			display.Icon(c, icon, display.IconProps{Size: display.IconSmall})

			mygo.Text(c, label)
		})
	})
	return button
}

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

func (a *App) buildInfoRow(c *mygo.Context, icon *mygo.SVG, label, value string) {
	mygo.Row(c).Gap(8).AlignItems(mygo.Center).Children(func() {
		display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
		display.Text(c, label, display.TextProps{Type: display.TextSecondary, FontSize: 13})
		mygo.Spacer(c).Grow(1).Basis(0)
		display.Text(c, value, display.TextProps{FontSize: 13})
	})
}

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

func (a *App) boardSize(c *mygo.Context) float32 {
	width, height := c.Size()
	return boardSizeFor(width, height)
}

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
			a.buildHistoryHeader(c)
			a.list.FollowEnd = true
			last := game != nil && game.LastMove != nil
			mygo.List(c, &a.list, len(history), func(i int) {
				a.buildHistoryRow(c, history[i], last && history[i].Number == game.LastMove.Number, tk)
			}).Grow(1).Basis(0)
		})
	})
}

func (a *App) buildHistoryHeader(c *mygo.Context) {
	mygo.Column(c).Gap(6).Children(func() {
		mygo.Row(c).Gap(historyColumnGap).AlignItems(mygo.Center).Padding(0, 6).Children(func() {
			historyCell(c, historyNumberWidth, "序号", display.TextProps{Type: display.TextSecondary, FontSize: 12})
			historyCell(c, historyPlayerWidth, "棋手", display.TextProps{Type: display.TextSecondary, FontSize: 12})
			historyCell(c, historyPlaceWidth, "位置", display.TextProps{Type: display.TextSecondary, FontSize: 12})
		})
		layout.Divider(c, layout.DividerProps{})
	})
}

func (a *App) buildHistoryRow(c *mygo.Context, move gomoku.Move, last bool, tk *theme.Tokens) {
	row := mygo.Row(c).Gap(historyColumnGap).AlignItems(mygo.Center).Padding(3, 6).Radius(6).Children(func() {
		historyCell(c, historyNumberWidth, fmt.Sprintf("%d", move.Number),
			display.TextProps{Type: display.TextSecondary, FontSize: 12})
		player := mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
			a.buildStoneDot(c, move.Player, tk)
			display.Text(c, playerLabel(move.Player), display.TextProps{FontSize: 13})
		})
		player.Width(historyPlayerWidth)
		historyCell(c, historyPlaceWidth, placeLabel(move.Row, move.Col), display.TextProps{FontSize: 13})
	})
	if last {
		row.Background(tk.FillSecondary)
	}
}

func historyCell(c *mygo.Context, width float32, text string, props display.TextProps) {
	cell := display.Text(c, text, props)
	cell.Width(width)
}

func placeLabel(row, col int) string {
	if row < 0 || row >= len(rowLabels) || col < 0 || col >= len(columnLabels) {
		return "-"
	}
	return columnLabels[col] + rowLabels[row]
}

func (a *App) buildStoneDot(c *mygo.Context, player gomoku.Player, tk *theme.Tokens) {
	dot := mygo.Box(c).Size(12, 12).Radius(6).Background(stoneColor(player)).Shrink(0)
	if player == gomoku.PlayerWhite {
		dot.Border(1, tk.Border)
	}
}

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

func sideLabel(config gomoku.PlayerConfig) string {
	if config.IsAI() {
		return "AI · " + config.EngineName()
	}
	return "玩家"
}

func executionLabel(game *gomoku.Game) string {
	if game.Status == gomoku.StatusFinished {
		return "已结束"
	}
	if game.Execution == gomoku.ExecutionPaused {
		return "已暂停"
	}
	return "进行中"
}
