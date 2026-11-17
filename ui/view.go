package ui

import (
	"fmt"
	"sync"

	mygo "github.com/egoist/mygo/ui"
	"github.com/lobehub/lobe-mygo/pkg/display"
	"github.com/lobehub/lobe-mygo/pkg/feedback"
	"github.com/lobehub/lobe-mygo/pkg/layout"
	"github.com/lobehub/lobe-mygo/pkg/lucide"
	"github.com/lobehub/lobe-mygo/pkg/theme"

	"mygo/assets"
	"mygo/internal/domain/gomoku"
)

const (
	sidebarWidth  = 296.0
	historyWidth  = 256.0
	pagePadding   = 16.0
	columnGap     = 16.0
	toolbarHeight = 56.0
	boardPadding  = 12.0
)

const (
	historyColumnGap   = 8.0
	historyNumberWidth = 38.0
	historyPlayerWidth = 70.0
	historyPlaceWidth  = 52.0
)

var modeNames = []string{"人机对战", "AI 对战", "玩家对战"}

var modeValues = []gomoku.Mode{gomoku.ModeHumanVsAI, gomoku.ModeAIVsAI, gomoku.ModeHumanVsHuman}

func (a *App) build(c *mygo.Context) {
	theme.Apply(c)
	tk := theme.TokensOf(c)
	a.board.applyTheme(tk)
	p := paletteOf(tk)

	page := mygo.Column(c).Fill()
	if !c.Vibrancy() {
		page.Background(p.canvas)
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

var (
	iconOnce sync.Once
	iconSrc  mygo.ImageSource
)

func brandMark(c *mygo.Context) {
	iconOnce.Do(func() {
		iconSrc, _ = mygo.DecodeImage(assets.IconPNG)
	})
	if iconSrc == nil {
		return
	}
	mygo.Image(c, iconSrc).Size(30, 30).Shrink(0).Fit(mygo.Contain)
}

func (a *App) buildToolbar(c *mygo.Context, tk *theme.Tokens) {
	p := paletteOf(tk)
	mygo.Row(c).Height(toolbarHeight).Padding(0, pagePadding).Gap(12).AlignItems(mygo.Center).Children(func() {
		brandMark(c)
		mygo.Text(c, "五子棋").FontSize(19).FontWeight(700).TextColor(p.text)
		mygo.Spacer(c).Grow(1).Basis(0)
		a.buildThinking(c)
		a.buildStatus(c, tk)
	})
}

func (a *App) buildThinking(c *mygo.Context) {
	p := paletteOf(theme.TokensOf(c))
	a.mu.Lock()
	thinking := !a.st.reviewing && a.st.game != nil && a.st.game.AIThinking
	a.mu.Unlock()
	if !thinking {
		return
	}
	mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
		display.Icon(c, lucide.LoaderCircle(), display.IconProps{Size: display.IconSmall, Spin: true})
		mygo.Text(c, "AI 思考中").FontSize(14).FontWeight(500).TextColor(p.accent)
	})
}

func (a *App) buildStatus(c *mygo.Context, tk *theme.Tokens) {
	p := paletteOf(tk)
	a.mu.Lock()
	game := a.st.game
	status := a.st.status
	reviewing := a.st.reviewing
	reviewStep := a.st.reviewStep
	reviewTotal := len(a.st.reviewMoves)
	a.mu.Unlock()
	if reviewing {
		chip(c, fmt.Sprintf("回溯 · 第 %d / %d 手", reviewStep, reviewTotal), p.warning)
		return
	}
	if game == nil {
		mygo.Text(c, "选择模式并开始一局新棋").FontSize(14).TextColor(p.faint)
		return
	}
	dot := p.accent
	switch game.Status {
	case gomoku.StatusFinished:
		dot = p.success
	}
	chip(c, status, dot)
}

func (a *App) buildSidebar(c *mygo.Context, tk *theme.Tokens) {
	_, height := c.Size()
	available := height - toolbarHeight - pagePadding
	column := mygo.Column(c).Width(sidebarWidth).MinHeight(0)
	column.Children(func() {
		scroll := layout.ScrollArea(c, layout.ScrollAreaProps{Direction: layout.Vertical}, func() {
			mygo.Column(c).Gap(columnGap).Width(sidebarWidth - 8).MinHeight(available).Children(func() {
				card := panel(c)
				card.Children(func() {
					mygo.Column(c).Gap(12).Children(func() {
						groupLabel(c, "新对局")
						a.buildModePicker(c)
						a.buildEnginePicker(c)
						a.buildNewGameButton(c)
						a.buildUndoButton(c)
						a.buildPauseButton(c)
						a.buildResult(c)
						a.buildNotice(c)
					})
				})
				a.buildGameInfo(c, tk)
				a.buildReplayPanel(c, tk)
			})
		})
		scroll.Grow(1).Basis(0)
	})
}

func (a *App) buildReplayPanel(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	games := a.st.replayGames
	reviewing := a.st.reviewing
	a.mu.Unlock()

	card := panel(c).Grow(1).Basis(0).MinHeight(160)
	card.Children(func() {
		mygo.Column(c).Gap(8).Grow(1).Basis(0).MinHeight(0).Children(func() {
			mygo.Row(c).AlignItems(mygo.Center).Children(func() {
				groupLabel(c, "棋局记录")
				mygo.Spacer(c).Grow(1).Basis(0)
				if reviewing {
					exit := ghostButton(c, lucide.ChevronLeft(), "退出回溯")
					if exit.Clicked() {
						a.stopReview()
					}
				}
			})
			if len(games) == 0 {
				emptyHint(c, "暂无棋局记录")
				return
			}
			mygo.List(c, &a.replayList, len(games), func(i int) {
				a.buildReplayRow(c, games[i], tk)
			}).Grow(1).Basis(0)
		})
	})
}

func (a *App) buildReplayRow(c *mygo.Context, game *gomoku.Game, tk *theme.Tokens) {
	p := paletteOf(tk)
	a.mu.Lock()
	active := a.st.reviewing && a.st.reviewGame != nil && a.st.reviewGame.ID == game.ID
	a.mu.Unlock()

	row := mygo.Row(c).Gap(8).AlignItems(mygo.Center).Padding(7, 8).Radius(8).Cursor(mygo.CursorPointer)
	row.OnClick(func() { a.startReview(game) })
	if active {
		row.Background(p.surfaceAlt)
	}
	row.Children(func() {
		mygo.Column(c).Gap(2).Children(func() {
			mygo.Text(c, modeLabel(game.Mode)).FontSize(14).FontWeight(500).TextColor(p.text)
			mygo.Text(c, replaySummary(game)).FontSize(13).TextColor(p.faint)
		})
		mygo.Spacer(c).Grow(1).Basis(0)
		label, color := replayResult(game)
		tinyTag(c, label, tagColor(color))
	})
}

func replaySummary(game *gomoku.Game) string {
	return fmt.Sprintf("共 %d 手", game.MoveCount)
}

func replayResult(game *gomoku.Game) (string, display.TagColor) {
	if game.Status != gomoku.StatusFinished {
		return "未完成", display.TagGray
	}
	switch game.Result {
	case gomoku.ResultBlackWin:
		return "黑胜", display.TagGray
	case gomoku.ResultWhiteWin:
		return "白胜", display.TagBlue
	case gomoku.ResultDraw:
		return "平局", display.TagGold
	}
	return "已结束", display.TagGray
}

func (a *App) buildModePicker(c *mygo.Context) {
	a.mu.Lock()
	index := a.st.modeIndex
	a.mu.Unlock()
	fieldLabel(c, "对战模式")
	segmented(c, &index, modeNames)
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
	fieldLabel(c, label)
	names := a.engineNames()
	if len(names) == 0 {
		mygo.Text(c, "没有可用引擎").FontSize(14).TextColor(paletteOf(theme.TokensOf(c)).danger)
		return
	}
	index := current() % len(names)
	segmented(c, &index, names)
	store(index % len(names))
}

func (a *App) buildAIEnginePickers(c *mygo.Context) {
	names := a.engineNames()
	if len(names) == 0 {
		mygo.Text(c, "没有可用引擎").FontSize(14).TextColor(paletteOf(theme.TokensOf(c)).danger)
		return
	}
	a.mu.Lock()
	first := a.st.firstEngineIndex % len(names)
	second := a.st.secondEngineIndex % len(names)
	a.mu.Unlock()

	fieldLabel(c, "先手（黑方）")
	segmented(c, &first, names)
	fieldLabel(c, "后手（白方）")
	segmented(c, &second, names)

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
	fieldLabel(c, "AI 执子")
	sides := []string{"黑方", "白方"}
	index := 1
	if black {
		index = 0
	}
	segmented(c, &index, sides)
	a.mu.Lock()
	a.st.aiIsBlack = index == 0
	a.mu.Unlock()
}

func (a *App) buildNewGameButton(c *mygo.Context) {
	button := primaryButton(c, lucide.Plus(), "开始新棋局")
	if button.Clicked() {
		a.newGame()
	}
}

func (a *App) buildUndoButton(c *mygo.Context) {
	a.mu.Lock()
	game := a.st.game
	reviewing := a.st.reviewing
	a.mu.Unlock()
	if reviewing || game == nil || game.Mode == gomoku.ModeAIVsAI || game.MoveCount == 0 {
		return
	}
	button := ghostButton(c, lucide.Undo2(), "悔棋")
	if button.Clicked() {
		a.undo()
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
	button := ghostButton(c, icon, label)
	if button.Clicked() {
		a.togglePause()
	}
}

func (a *App) buildResult(c *mygo.Context) {
	p := paletteOf(theme.TokensOf(c))
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()
	if game == nil || game.Status != gomoku.StatusFinished {
		return
	}
	color := p.success
	icon := lucide.Trophy()
	title := "本局结束"
	switch game.Result {
	case gomoku.ResultBlackWin:
		title = "黑棋获胜"
	case gomoku.ResultWhiteWin:
		title = "白棋获胜"
	case gomoku.ResultDraw:
		title = "平局"
		color = p.warning
		icon = lucide.CircleCheck()
	}
	noticeRow(c, icon, title, color)
}

func iconButton(c *mygo.Context, icon *mygo.SVG, label string) mygo.Element {
	p := paletteOf(theme.TokensOf(c))
	button := mygo.ButtonBase(c).Height(34).PaddingX(10).Radius(8).
		Background(p.surface).Border(1, p.border).Cursor(mygo.CursorPointer)
	if button.Hovered() {
		button.Border(1, p.borderStrong)
	}
	button.Children(func() {
		mygo.Row(c).Gap(5).AlignItems(mygo.Center).Children(func() {
			display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
			mygo.Text(c, label).FontSize(14).TextColor(p.muted)
		})
	})
	return button
}

func (a *App) buildNotice(c *mygo.Context) {
	p := paletteOf(theme.TokensOf(c))
	a.mu.Lock()
	kind := a.st.noticeKind
	notice := a.st.notice
	a.mu.Unlock()
	if kind == noticeNone || notice == "" {
		return
	}
	color := p.accent
	icon := lucide.Info()
	if kind == noticeError {
		color = p.danger
		icon = lucide.CircleAlert()
	}
	noticeRow(c, icon, notice, color)
}

func (a *App) buildGameInfo(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()

	card := panel(c)
	card.Children(func() {
		mygo.Column(c).Gap(10).Children(func() {
			groupLabel(c, "对局信息")
			if game == nil {
				emptyHint(c, "还没有进行中的棋局")
				return
			}
			a.buildInfoRow(c, lucide.Target(), "模式", modeLabel(game.Mode))
			a.buildInfoRow(c, lucide.Circle(), "黑棋", sideLabel(game.Black()))
			a.buildInfoRow(c, lucide.Circle(), "白棋", sideLabel(game.White()))
			a.buildInfoRow(c, lucide.Dices(), "手数", fmt.Sprintf("%d 手", game.MoveCount))
			a.buildInfoRow(c, lucide.Activity(), "状态", executionLabel(game))
		})
	})
}

func (a *App) buildInfoRow(c *mygo.Context, icon *mygo.SVG, label, value string) {
	p := paletteOf(theme.TokensOf(c))
	mygo.Row(c).Gap(8).AlignItems(mygo.Center).TextColor(p.faint).Children(func() {
		display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
		mygo.Text(c, label).FontSize(14).TextColor(p.muted)
		mygo.Spacer(c).Grow(1).Basis(0)
		mygo.Text(c, value).FontSize(14).FontWeight(500).TextColor(p.text)
	})
}

func (a *App) buildBoardArea(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	game := a.st.game
	reviewing := a.st.reviewing
	reviewGame := a.st.reviewGame
	reviewMoves := a.st.reviewMoves
	reviewStep := a.st.reviewStep
	a.mu.Unlock()

	if reviewing && reviewGame != nil {
		a.board.Game = replayPosition(reviewGame, reviewMoves, reviewStep)
		a.board.ReadOnly = true
		a.board.OnPick = nil
	} else {
		a.board.Game = game
		a.board.ReadOnly = false
		a.board.OnPick = a.pick
	}

	card := mygo.Box(c).Grow(1).Basis(0).Padding(boardPadding)
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
	horizontal := width - sidebarWidth - historyWidth - 2*columnGap - 2*pagePadding - 2*boardPadding
	vertical := height - toolbarHeight - 2*pagePadding - 2*boardPadding
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
	reviewing := a.st.reviewing
	reviewMoves := a.st.reviewMoves
	reviewStep := a.st.reviewStep
	reviewTotal := len(a.st.reviewMoves)
	a.mu.Unlock()

	if reviewing {
		history = reviewMoves
	}

	card := panel(c).Width(historyWidth)
	card.Children(func() {
		mygo.Column(c).Gap(10).Grow(1).Basis(0).Children(func() {
			groupLabel(c, "落子记录")
			if reviewing {
				a.buildReviewControls(c, reviewStep, reviewTotal)
			}
			if len(history) == 0 {
				emptyHint(c, "还没有落子")
				return
			}
			a.buildHistoryHeader(c)
			if reviewing {
				a.reviewList.FollowEnd = false
				mygo.List(c, &a.reviewList, len(history), func(i int) {
					a.buildHistoryRow(c, history[i], i == reviewStep-1, i+1, tk)
				}).Grow(1).Basis(0)
				return
			}
			a.list.FollowEnd = true
			last := game != nil && game.LastMove != nil
			mygo.List(c, &a.list, len(history), func(i int) {
				a.buildHistoryRow(c, history[i], last && history[i].Number == game.LastMove.Number, -1, tk)
			}).Grow(1).Basis(0)
		})
	})
}

func (a *App) buildReviewControls(c *mygo.Context, step, total int) {
	p := paletteOf(theme.TokensOf(c))
	mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
		prev := iconButton(c, lucide.ChevronLeft(), "上一手")
		if prev.Clicked() {
			a.stepReview(-1)
		}
		mygo.Text(c, fmt.Sprintf("%d/%d", step, total)).
			FontSize(13).FontWeight(600).TextColor(p.accent)
		next := iconButton(c, lucide.ChevronRight(), "下一手")
		if next.Clicked() {
			a.stepReview(1)
		}
	})
}

func (a *App) buildHistoryHeader(c *mygo.Context) {
	p := paletteOf(theme.TokensOf(c))
	mygo.Column(c).Gap(6).Children(func() {
		mygo.Row(c).Gap(historyColumnGap).AlignItems(mygo.Center).Padding(0, 6).Children(func() {
			historyCell(c, historyNumberWidth, "序号", p.faint, 13, 600)
			historyCell(c, historyPlayerWidth, "棋手", p.faint, 13, 600)
			historyCell(c, historyPlaceWidth, "位置", p.faint, 13, 600)
		})
		mygo.Box(c).Height(1).Background(p.border).AlignSelf(mygo.Stretch)
	})
}

func (a *App) buildHistoryRow(c *mygo.Context, move gomoku.Move, highlight bool, step int, tk *theme.Tokens) {
	p := paletteOf(tk)
	row := mygo.Row(c).Gap(historyColumnGap).AlignItems(mygo.Center).Padding(4, 6).Radius(7).Children(func() {
		historyCell(c, historyNumberWidth, fmt.Sprintf("%d", move.Number), p.faint, 13, 500)
		player := mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
			a.buildStoneDot(c, move.Player, tk)
			mygo.Text(c, playerLabel(move.Player)).FontSize(14).TextColor(p.text)
		})
		player.Width(historyPlayerWidth)
		historyCell(c, historyPlaceWidth, placeLabel(move.Row, move.Col), p.muted, 14, 500)
	})
	if step > 0 {
		row.Cursor(mygo.CursorPointer).OnClick(func() { a.setReviewStep(step) })
	}
	if highlight {
		row.Background(p.accent.Alpha(0.10))
	}
}

func historyCell(c *mygo.Context, width float32, text string, color mygo.Color, size float32, weight int) {
	cell := mygo.Text(c, text).FontSize(size).FontWeight(weight).TextColor(color).SingleLine()
	cell.Width(width)
}

func placeLabel(row, col int) string {
	if row < 0 || row >= len(rowLabels) || col < 0 || col >= len(columnLabels) {
		return "-"
	}
	return columnLabels[col] + rowLabels[row]
}

func (a *App) buildStoneDot(c *mygo.Context, player gomoku.Player, tk *theme.Tokens) {
	light, _ := stoneColors(player == gomoku.PlayerBlack, tk)
	p := paletteOf(tk)
	mygo.Box(c).Size(12, 12).Radius(6).Background(light).Shrink(0).Border(1, p.borderStrong)
}

func modeLabel(mode gomoku.Mode) string {
	switch mode {
	case gomoku.ModeAIVsAI:
		return "AI 对战"
	case gomoku.ModeHumanVsAI:
		return "人机对战"
	case gomoku.ModeHumanVsHuman:
		return "玩家对战"
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
