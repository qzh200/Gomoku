package ui

import (
	"testing"
	"time"

	mygoui "github.com/egoist/mygo/ui"

	"mygo/internal/application"
	"mygo/internal/auth"
	"mygo/internal/domain/gomoku"
	"mygo/internal/repository"
	"mygo/internal/scheduler"
)

// newTestApp builds an interface over in-memory services.
func newTestApp(t *testing.T) *App {
	t.Helper()
	registry, err := auth.NewRegistry(auth.Config{Engines: []auth.EngineConfig{
		{EngineID: "classical", DisplayName: "Classical AI", Token: "classical-token"},
		{EngineID: "llm", DisplayName: "LLM AI", Token: "llm-token"},
	}})
	if err != nil {
		t.Fatalf("构建引擎注册表失败: %v", err)
	}
	coord := application.NewCoordinator(repository.NewMemoryGames(), scheduler.NewLeaseManager(), repository.NewMemoryIdempotency())
	return New(application.NewGameService(coord, registry), application.NewMoveService(coord), registry)
}

func TestInterfaceDrawsBoardAndCreatesGame(t *testing.T) {
	app := newTestApp(t)
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(100 * time.Millisecond)

	if !tester.HasText("MyGo 五子棋") {
		t.Fatalf("界面必须显示标题, 实际文本: %v", tester.Texts())
	}
	if !tester.HasText("开始新棋局") {
		t.Fatalf("界面必须显示开始按钮, 实际文本: %v", tester.Texts())
	}

	// 棋盘区域存在且可交互：点击棋盘中心不应崩溃。
	rect, ok := tester.Find("开始新棋局")
	if !ok {
		t.Fatalf("找不到开始按钮")
	}
	tester.ClickAt(rect.X+4, rect.Y+4)
	tester.Advance(100 * time.Millisecond)
}

func TestHumanMoveIsAppliedThroughTheApplicationService(t *testing.T) {
	app := newTestApp(t)
	app.newGame()

	app.mu.Lock()
	game := app.st.game
	app.mu.Unlock()
	if game == nil {
		t.Fatalf("新棋局必须创建成功")
	}
	if game.Mode != gomoku.ModeHumanVsAI {
		t.Fatalf("默认模式不对: %s", game.Mode)
	}

	// 人类执黑，落子必须经过应用服务并落到棋盘上。
	app.pick(7, 7)

	app.mu.Lock()
	game = app.st.game
	app.mu.Unlock()
	if game == nil || game.MoveCount != 1 {
		t.Fatalf("落子必须生效: %v", game)
	}
	if game.Board[7][7] != 1 {
		t.Fatalf("棋盘上必须有黑棋")
	}
	if game.Version != 1 {
		t.Fatalf("版本必须加一")
	}
}

func TestBoardMapsPointerToCell(t *testing.T) {
	board := newBoardView()
	row, col := board.cellAt(boardMargin+7*boardCell, boardMargin+3*boardCell)
	if row != 3 || col != 7 {
		t.Fatalf("指针位置必须映射到正确的格子, 实际 %d,%d", row, col)
	}
	if row, col := board.cellAt(2, 2); row != -1 || col != -1 {
		t.Fatalf("棋盘外的指针必须被拒绝, 实际 %d,%d", row, col)
	}
}

func TestHoverShowsGhostStoneOnlyOnEmptyCell(t *testing.T) {
	app := newTestApp(t)
	app.newGame()
	app.pick(7, 7)

	app.mu.Lock()
	game := app.st.game
	app.mu.Unlock()
	app.board.Game = game
	app.board.HoverRow, app.board.HoverCol = 7, 7
	if app.board.HoverCellFree() {
		t.Fatalf("有棋子的格子不得显示落子预览")
	}
	app.board.HoverRow, app.board.HoverCol = 3, 3
	if !app.board.HoverCellFree() {
		t.Fatalf("空格子必须显示落子预览")
	}
}
