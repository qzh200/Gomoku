package ui

import (
	"strings"
	"testing"
	"time"

	mygoui "github.com/egoist/mygo/ui"

	"mygo/internal/domain/gomoku"
)

func TestClickOnBoardPlacesAStone(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()

	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)

	geo := geometryOf(app.board.rect.W)
	rect := app.board.rect
	if rect.W == 0 {
		t.Fatalf("棋盘没有绘制")
	}

	col, row := 4, 6
	x := rect.X + geo.OriginX + float32(col)*geo.Cell
	y := rect.Y + geo.OriginY + float32(row)*geo.Cell
	tester.ClickAt(x, y)
	tester.Advance(200 * time.Millisecond)

	app.mu.Lock()
	game := app.st.game
	app.mu.Unlock()
	if game == nil || game.MoveCount != 1 {
		t.Fatalf("点击棋盘必须落子: %v", game)
	}
	if game.Board[row][col] != 1 {
		t.Fatalf("棋子必须落在点击的格子 (%d,%d)", row, col)
	}
}

func TestClickOutsideTheGridIsIgnored(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()

	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)

	geo := geometryOf(app.board.rect.W)
	rect := app.board.rect

	tester.ClickAt(rect.X+2, rect.Y+2)
	tester.Advance(200 * time.Millisecond)

	app.mu.Lock()
	game := app.st.game
	app.mu.Unlock()
	if game == nil || game.MoveCount != 0 {
		t.Fatalf("棋盘外的点击不得落子: %v", game)
	}
	_ = geo
}

func TestHoverPreviewFollowsThePointer(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()

	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)

	geo := geometryOf(app.board.rect.W)
	rect := app.board.rect
	tester.Move(rect.X+geo.OriginX+8*geo.Cell, rect.Y+geo.OriginY+2*geo.Cell)
	tester.Advance(200 * time.Millisecond)

	if app.board.HoverRow != 2 || app.board.HoverCol != 8 {
		t.Fatalf("悬停格子不对: %d,%d", app.board.HoverRow, app.board.HoverCol)
	}
	if !app.board.hoverCellFree() {
		t.Fatalf("空格子必须可以预览")
	}
}

func TestAISidePickerChoosesTheSide(t *testing.T) {
	app := newTestApp(t)
	app.st.aiIsBlack = true
	app.newGame()

	app.mu.Lock()
	game := app.st.game
	app.mu.Unlock()
	if game == nil {
		t.Fatalf("新棋局必须创建成功")
	}
	if game.Black().EngineName() != "classical" {
		t.Fatalf("AI 必须执黑: %v", game.Black())
	}
	if game.White().EngineName() != "" {
		t.Fatalf("白棋必须是人类: %v", game.White())
	}
	if game.Current != gomoku.PlayerBlack {
		t.Fatalf("AI 执黑时第一回合属于黑棋")
	}
}

func TestAIGameUsesTheChosenFirstEngine(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 1
	app.st.firstEngineIndex = 1
	app.st.secondEngineIndex = 0
	app.newGame()

	app.mu.Lock()
	game := app.st.game
	app.mu.Unlock()
	if game == nil {
		t.Fatalf("新棋局必须创建成功")
	}
	if game.Black().EngineName() != "llm" {
		t.Fatalf("先手引擎必须执黑: %v", game.Black())
	}
	if game.White().EngineName() != "classical" {
		t.Fatalf("后手引擎必须执白: %v", game.White())
	}
	if game.Current != gomoku.PlayerBlack {
		t.Fatalf("黑棋必须先手")
	}
}

func TestStartButtonStartsAGame(t *testing.T) {
	app := newTestApp(t)
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)

	if err := tester.Click("开始新棋局"); err != nil {
		t.Fatalf("点击开始按钮失败: %v", err)
	}
	tester.Advance(200 * time.Millisecond)

	app.mu.Lock()
	game := app.st.game
	app.mu.Unlock()
	if game == nil {
		t.Fatalf("点击开始按钮必须创建棋局")
	}
}

func TestButtonsShowTheirLabelOnce(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 1
	app.newGame()
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)

	counts := map[string]int{}
	for _, text := range tester.Texts() {
		counts[text]++
	}
	for _, label := range []string{"开始新棋局", "暂停对战"} {
		if counts[label] != 1 {
			t.Fatalf("按钮 %q 必须只出现一次, 实际 %d 次", label, counts[label])
		}
	}
}

func TestHistoryShowsEveryMove(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()
	for _, move := range [][2]int{{7, 7}, {7, 8}, {6, 6}} {
		app.pick(move[0], move[1])
	}
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)

	texts := tester.Texts()
	joined := strings.Join(texts, "|")
	for _, want := range []string{"H8", "I8", "G7"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("落子记录必须显示坐标 %s, 实际 %v", want, texts)
		}
	}
}

func TestReplayRewindsAPastGame(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()
	for _, move := range [][2]int{{7, 7}, {7, 8}, {6, 6}, {8, 8}, {5, 5}, {5, 6}, {4, 4}, {4, 5}, {3, 3}} {
		app.pick(move[0], move[1])
	}
	past := app.games.List()[0]
	app.newGame()

	app.mu.Lock()
	for _, game := range app.st.replayGames {
		if game.Status != gomoku.StatusFinished {
			app.mu.Unlock()
			t.Fatalf("回溯列表只能列出已结束的棋局, 实际 %s 仍在进行", game.ID)
		}
	}
	app.mu.Unlock()

	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)

	if !tester.HasText("棋局回溯") {
		t.Fatalf("界面必须有棋局回溯面板, 实际 %v", tester.Texts())
	}
	if tester.HasText(past.ID) {
		t.Fatalf("界面不能显示棋局编号 %q", past.ID)
	}

	app.startReview(past)
	tester.Frame()
	if !tester.HasText("回溯 · 第 9 / 9 手") {
		t.Fatalf("回溯必须停在末手, 实际 %v", tester.Texts())
	}
	if app.board.Game == nil || app.board.Game.MoveCount != 9 {
		t.Fatalf("回溯棋盘必须还原 9 手")
	}

	app.setReviewStep(1)
	tester.Frame()
	if app.board.Game == nil || app.board.Game.MoveCount != 1 || app.board.Game.Board[7][8] != 0 {
		t.Fatalf("回溯到第 1 手时棋盘只能保留 1 个棋子")
	}

	app.stepReview(-5)
	tester.Frame()
	if app.board.Game == nil || app.board.Game.MoveCount != 0 {
		t.Fatalf("回溯不能越过开局")
	}

	app.stopReview()
	tester.Frame()
	app.mu.Lock()
	live := app.st.game
	app.mu.Unlock()
	if live == nil || live.MoveCount != 0 {
		t.Fatalf("退出回溯后必须回到当前对局")
	}
}
