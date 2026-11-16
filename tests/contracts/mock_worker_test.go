package contracts

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestMockWorkerFinishesAGame(t *testing.T) {
	python, err := exec.LookPath("python")
	if err != nil {
		t.Skip("找不到 python，跳过 mock worker 测试")
	}
	h := newHarness(t)
	h.createGame("ai_vs_ai", aiVsAIPlayers())

	workers := []struct{ engine, id string }{
		{"classical", "classical-test-worker"},
		{"llm", "llm-test-worker"},
	}
	for _, worker := range workers {
		cmd := exec.Command(python, filepath.Join("..", "..", "mock_worker", "mock_worker.py"),
			"--url", h.server.URL+"/api/v1",
			"--engine", worker.engine,
			"--token", engineTokens[worker.engine],
			"--worker-id", worker.id,
		)
		output := &strings.Builder{}
		cmd.Stdout = output
		cmd.Stderr = output
		if err := cmd.Start(); err != nil {
			t.Skipf("无法启动 mock worker: %v", err)
		}
		t.Cleanup(func() {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
		})
	}

	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		status, data := h.do("GET", "/api/v1/games/game_001", nil, nil)
		if status != 200 {
			t.Fatalf("查询棋局失败: %d %s", status, data)
		}
		game := decode(t, data)
		if game["status"] == "finished" {
			if game["result"] == "ongoing" {
				t.Fatalf("结束的棋局必须有结果")
			}
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("mock worker 未能在时限内完成整局")
}
