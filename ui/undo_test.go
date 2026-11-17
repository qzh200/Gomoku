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
