package contracts

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestClassicalWorkerFinishesAGame runs the real classical worker for both
// sides of an AI game and checks that the game reaches its end.
//
// The worker is started twice, once per engine: the algorithm does not depend
// on which engine the credential names.
func TestClassicalWorkerFinishesAGame(t *testing.T) {
	python, err := exec.LookPath("python")
	if err != nil {
		t.Skip("找不到 python，跳过 classical worker 测试")
	}
	root := filepath.Join("..", "..")
	if _, err := os.Stat(filepath.Join(root, "classical_worker", "__main__.py")); err != nil {
		t.Skip("找不到 classical_worker 包，跳过 classical worker 测试")
	}
	h := newHarness(t)
	h.createGame("ai_vs_ai", aiVsAIPlayers())

	outputs := make([]*strings.Builder, 0, 2)
	for _, engine := range []string{"classical", "llm"} {
		output := &strings.Builder{}
		outputs = append(outputs, output)
		cmd := exec.Command(python, "-m", "classical_worker")
		cmd.Dir = root
		cmd.Env = append(os.Environ(),
			"MYGO_URL="+h.server.URL+"/api/v1",
			"MYGO_ENGINE_TOKEN="+engineTokens[engine],
			"MYGO_ENGINE_ID="+engine,
			"MYGO_WORKER_ID="+engine+"-classical-worker",
			"CLASSICAL_DEPTH=2",
			"CLASSICAL_WIDTH=8",
			"CLASSICAL_TIME_BUDGET=0.3",
		)
		cmd.Stdout = output
		cmd.Stderr = output
		if err := cmd.Start(); err != nil {
			t.Skipf("无法启动 classical worker: %v", err)
		}
		t.Cleanup(func() {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
		})
	}

	deadline := time.Now().Add(150 * time.Second)
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
			if count, _ := game["move_count"].(float64); count < 9 {
				t.Fatalf("棋局必须真的下过，手数=%v", game["move_count"])
			}
			for _, output := range outputs {
				if !strings.Contains(output.String(), "落子成功") {
					t.Fatalf("两个 Worker 都必须落过子:\n%s", output.String())
				}
			}
			t.Logf("棋局结束，手数=%v 结果=%v", game["move_count"], game["result"])
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("classical worker 未能在时限内完成整局:\n%s\n%s", outputs[0].String(), outputs[1].String())
}
