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
		mygo.Row(c).Padding(16, 20).Gap(10).AlignItems(mygo.Center).Children(func() {
			display.Icon(c, lucide.Grid3x3(), display.IconProps{Size: display.IconMiddle})
			display.Text(c, "MyGo 五子棋", display.TextProps{Heading: 4})
			mygo.Spacer(c).Grow(1).Basis(0)
			a.buildStatus(c)
		})
		mygo.Row(c).Grow(1).Basis(0).MinHeight(0).Padding(0, 20, 20, 20).Gap(20).AlignItems(mygo.Stretch).Children(func() {
			a.buildSidebar(c, tk)
			a.buildBoardArea(c, tk)
			a.buildHistory(c, tk)
		})
		feedback.ToastHost(c, feedback.ToastHostProps{})
	})
}

// buildStatus shows the status line of the current game.
func (a *App) buildStatus(c *mygo.Context) {
	a.mu.Lock()
	status := a.st.status
	a.mu.Unlock()
	if status == "" {
		return
	}
	display.Tag(c, status, display.TagProps{Color: display.TagBlue, Round: true})
}

// buildSidebar holds the game controls.
func (a *App) buildSidebar(c *mygo.Context, tk *theme.Tokens) {
	panel := layout.Block(c, layout.BlockProps{Variant: layout.BlockFilled}).Width(300).Padding(16).Radius(tk.BorderRadiusLG)
	panel.Children(func() {
		mygo.Column(c).Gap(16).Children(func() {
			display.Text(c, "新对局", display.TextProps{Type: display.TextSecondary, Strong: true})
			a.buildModePicker(c)
			a.buildEnginePicker(c)
			a.buildNewGameButton(c)
			a.buildPauseButton(c)
			a.buildNotice(c)
		})
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
	a.mu.Unlock()
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

// buildNewGameButton starts a new game.
func (a *App) buildNewGameButton(c *mygo.Context) {
	button := action.Button(c, "开始新棋局", action.ButtonProps{Variant: action.Primary})
	button.Children(func() {
		mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
			display.Icon(c, lucide.Plus(), display.IconProps{Size: display.IconSmall})
			display.Text(c, "开始新棋局", display.TextProps{})
		})
	})
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
	button := action.Button(c, label, action.ButtonProps{Variant: action.Default})
	button.Children(func() {
		mygo.Row(c).Gap(6).AlignItems(mygo.Center).Children(func() {
			display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
			display.Text(c, label, display.TextProps{})
		})
	})
	if button.Clicked() {
		a.togglePause()
	}
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

// buildBoardArea draws the board and its frame.
func (a *App) buildBoardArea(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	game := a.st.game
	a.mu.Unlock()
	a.board.Game = game
	a.board.OnPick = a.pick

	card := layout.Block(c, layout.BlockProps{Variant: layout.BlockFilled}).Grow(1).Basis(0).Padding(16).Radius(tk.BorderRadiusLG)
	card.Children(func() {
		mygo.Box(c).Grow(1).Basis(0).Center().Children(func() {
			a.board.View(c)
		})
	})
}

// buildHistory lists the moves of the current game.
func (a *App) buildHistory(c *mygo.Context, tk *theme.Tokens) {
	a.mu.Lock()
	history := a.st.history
	a.mu.Unlock()

	panel := layout.Block(c, layout.BlockProps{Variant: layout.BlockFilled}).Width(260).Padding(16).Radius(tk.BorderRadiusLG)
	panel.Children(func() {
		mygo.Column(c).Gap(8).Children(func() {
			display.Text(c, "落子记录", display.TextProps{Type: display.TextSecondary, Strong: true})
			if len(history) == 0 {
				display.Text(c, "还没有落子", display.TextProps{Type: display.TextSecondary, FontSize: 12})
				return
			}
			layout.ScrollArea(c, layout.ScrollAreaProps{Direction: layout.Vertical, Fade: true}, func() {
				mygo.Column(c).Gap(6).Children(func() {
					for _, move := range history {
						mygo.Row(c).Gap(8).AlignItems(mygo.Center).Children(func() {
							mygo.Box(c).Size(10, 10).Radius(5).Background(stoneColor(move.Player)).Shrink(0)
							display.Text(c, fmt.Sprintf("#%d  %s  (%d, %d)", move.Number, playerLabel(move.Player), move.Row, move.Col),
								display.TextProps{Type: display.TextSecondary, FontSize: 12})
						})
					}
				})
			})
		})
	})
}
