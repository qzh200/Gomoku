package ui

import (
	"testing"
	"time"

	mygoui "github.com/egoist/mygo/ui"
)

func TestUndoTakesBackTheLastMove(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()
	for _, move := range [][2]int{{7, 7}, {7, 8}, {6, 6}} {
		app.pick(move[0], move[1])
	}
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)
	if !tester.HasText("悔棋") {
		t.Fatalf("有落子时界面必须显示悔棋按钮, 实际 %v", tester.Texts())
	}

	app.undo()
	tester.Frame()
	app.mu.Lock()
	game := app.st.game
	app.mu.Unlock()
	if game == nil || game.MoveCount != 2 || game.Board[6][6] != 0 {
		t.Fatalf("悔棋必须撤销最后一手")
	}
	if !tester.HasText("H8") || !tester.HasText("I8") || tester.HasText("G7") {
		t.Fatalf("落子记录必须同步撤销最后一手, 实际 %v", tester.Texts())
	}

	app.undo()
	app.undo()
	tester.Frame()
	app.mu.Lock()
	game = app.st.game
	app.mu.Unlock()
	if game == nil || game.MoveCount != 0 {
		t.Fatalf("连续悔棋必须回到空棋盘")
	}
	if tester.HasText("悔棋") {
		t.Fatalf("没有落子时不应显示悔棋按钮")
	}
}

func TestUnfinishedOldGamesAreRecordedButNotLive(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()
	app.pick(7, 7)
	old := app.games.List()[0]
	app.newGame()

	app.mu.Lock()
	recorded := len(app.st.replayGames)
	firstID := ""
	if recorded > 0 {
		firstID = app.st.replayGames[0].ID
	}
	app.mu.Unlock()
	if recorded != 1 || firstID != old.ID {
		t.Fatalf("旧棋局必须全部记录, 实际 %d 局", recorded)
	}

	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)
	if !tester.HasText("未完成") {
		t.Fatalf("未完成的旧棋局必须标记为未完成, 实际 %v", tester.Texts())
	}
}

func TestNewGameDropsTheAbandonedEmptyGame(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()
	app.newGame()
	app.newGame()

	if games := app.games.List(); len(games) != 1 {
		t.Fatalf("未落子的旧棋局必须丢弃, 实际保留 %d 局", len(games))
	}
	app.mu.Lock()
	recorded := len(app.st.replayGames)
	app.mu.Unlock()
	if recorded != 0 {
		t.Fatalf("空棋局不应进入棋局记录, 实际 %d 条", recorded)
	}
}

func TestNewGameFromReviewKeepsOnlyRealRecords(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()
	for _, move := range [][2]int{{7, 7}, {7, 8}, {6, 6}, {8, 8}, {5, 5}, {5, 6}, {4, 4}, {4, 5}, {3, 3}} {
		app.pick(move[0], move[1])
	}
	app.newGame()
	past := app.games.List()[0]
	app.startReview(past)
	app.newGame()

	app.mu.Lock()
	defer app.mu.Unlock()
	if len(app.st.replayGames) != 1 || app.st.replayGames[0].ID != past.ID {
		t.Fatalf("回溯中开新局不应新增记录, 实际 %d 条", len(app.st.replayGames))
	}
	if app.st.reviewing {
		t.Fatalf("开始新棋局必须退出回溯")
	}
}
