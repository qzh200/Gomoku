package ui

import (
	"strings"
	"testing"
	"time"

	mygoui "github.com/egoist/mygo/ui"

	"mygo/internal/domain/gomoku"
)

// TestClickOnBoardPlacesAStone drives the board through the tester and checks
// that a click lands on the cell the pointer is over.
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
	// 点击 E7：第五列，第七行。
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

// TestClickOutsideTheGridIsIgnored checks that a click outside the grid does
// not place a stone.
func TestClickOutsideTheGridIsIgnored(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()

	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)

	geo := geometryOf(app.board.rect.W)
	rect := app.board.rect
	// 棋盘的最左上是标签区，不是格子。
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

// TestHoverPreviewFollowsThePointer checks that the hovered cell is tracked.
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

// TestAISidePickerChoosesTheSide checks that the interface can start a game
// where the AI takes the black stones.
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

// TestAIGameUsesTheChosenFirstEngine checks that an AI game gives the black
// stones to the engine picked as the first mover.
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

// TestStartButtonStartsAGame clicks the button in the sidebar and checks that
// a game starts.
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

// TestButtonsShowTheirLabelOnce checks that a button does not render its
// label twice.
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

// TestHistoryShowsEveryMove checks that the move list holds every move of the
// game, with its number and its place.
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
