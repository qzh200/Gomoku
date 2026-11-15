package contracts

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"mygo/internal/application"
	"mygo/internal/auth"
	"mygo/internal/domain/gomoku"
	"mygo/internal/repository"
	"mygo/internal/scheduler"
	httpapi "mygo/internal/transport/http"
)

// TestUIGameIsClaimableByAWorker checks that a game created through the
// application services, the way the interface creates it, is picked up by a
// Python worker over the API.
func TestUIGameIsClaimableByAWorker(t *testing.T) {
	python, err := exec.LookPath("python")
	if err != nil {
		t.Skip("找不到 python，跳过测试")
	}
	cfg := auth.Config{
		LocalClientToken: "local-test-token",
		Engines: []auth.EngineConfig{
			{EngineID: "classical", DisplayName: "Classical AI", Token: engineTokens["classical"]},
			{EngineID: "llm", DisplayName: "LLM AI", Token: engineTokens["llm"]},
		},
	}
	registry, err := auth.NewRegistry(cfg)
	if err != nil {
		t.Fatalf("构建引擎注册表失败: %v", err)
	}
	coord := application.NewCoordinator(repository.NewMemoryGames(), scheduler.NewLeaseManager(), repository.NewMemoryIdempotency())
	gameService := application.NewGameService(coord, registry)
	moveService := application.NewMoveService(coord)
	claimService := application.NewClaimService(coord)
	server := httptestServer(t, httpapi.NewServer(gameService, moveService, claimService, registry).Handler())

	// 界面用应用服务创建棋局：AI 执白。
	result, err := gameService.Create("", application.CreateGameRequest{
		Mode: gomoku.ModeHumanVsAI,
		Players: map[string]application.CreatePlayer{
			"black": {Type: gomoku.TypeHuman, Config: map[string]any{}},
			"white": {Type: gomoku.TypeAI, EngineID: "classical", Config: map[string]any{}},
		},
	}, "")
	if err != nil {
		t.Fatalf("创建棋局失败: %v", err)
	}
	gameID := result.Game.ID

	// 人类先落子，AI 的回合随后开放。
	human := application.Caller{Kind: application.CallerOwner, OwnerTokenHash: auth.HashToken(result.OwnerToken)}
	if _, err := moveService.Submit(gameID, human, application.SubmitMoveRequest{
		Row: 7, Col: 7, ExpectedVersion: 0,
	}); err != nil {
		t.Fatalf("人类落子失败: %v", err)
	}

	worker := exec.Command(python, filepath.Join("..", "..", "mock_worker", "mock_worker.py"),
		"--url", server+"/api/v1",
		"--engine", "classical",
		"--token", engineTokens["classical"],
		"--worker-id", "classical-ui-worker",
	)
	output := &strings.Builder{}
	worker.Stdout = output
	worker.Stderr = output
	if err := worker.Start(); err != nil {
		t.Skipf("无法启动 mock worker: %v", err)
	}
	defer func() {
		_ = worker.Process.Kill()
		_, _ = worker.Process.Wait()
	}()

	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		game, err := gameService.Get(gameID)
		if err != nil {
			t.Fatalf("查询棋局失败: %v", err)
		}
		if game.MoveCount >= 2 {
			if game.Board[7][8] != 0 && game.Board[7][8] != 1 {
				t.Fatalf("AI 的落子不在棋盘上")
			}
			if game.Version < 2 {
				t.Fatalf("棋局版本必须随 AI 的落子增加")
			}
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("worker 没有接手界面的棋局, 输出:\n%s", output.String())
}
