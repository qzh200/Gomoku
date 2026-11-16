package contracts

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestAZWorkerFinishesAGame runs the real AlphaZero worker for both sides of
// an AI game and checks that the game reaches its end.
//
// The worker needs PyTorch, so the test uses the virtual environment of the
// project and is skipped when it is missing.
func TestAZWorkerFinishesAGame(t *testing.T) {
	root := filepath.Join("..", "..")
	absolute, err := filepath.Abs(root)
	if err != nil {
		t.Fatalf("解析仓库根目录失败: %v", err)
	}
	root = absolute
	python := filepath.Join(root, ".venv", "Scripts", "python.exe")
	if _, err := os.Stat(python); err != nil {
		t.Skip("找不到 .venv，跳过 AlphaZero worker 测试")
	}
	if _, err := os.Stat(filepath.Join(root, "az_worker", "__main__.py")); err != nil {
		t.Skip("找不到 az_worker 包，跳过 AlphaZero worker 测试")
	}
	check := exec.Command(python, "-c", "import torch")
	check.Dir = root
	if err := check.Run(); err != nil {
		t.Skip("虚拟环境里没有 PyTorch，跳过 AlphaZero worker 测试")
	}

	h := newHarnessWithEngines(t, 0, "az")
	h.createGame("ai_vs_ai", map[string]any{
		"black": map[string]any{"type": "ai", "engine_id": "az"},
		"white": map[string]any{"type": "ai", "engine_id": "az"},
	})

	outputs := make([]*strings.Builder, 0, 2)
	for _, side := range []string{"black", "white"} {
		output := &strings.Builder{}
		outputs = append(outputs, output)
		cmd := exec.Command(python, "-m", "az_worker")
		cmd.Dir = root
		cmd.Env = append(os.Environ(),
			"MYGO_URL="+h.server.URL+"/api/v1",
			"MYGO_ENGINE_TOKEN="+engineTokens["az"],
			"MYGO_ENGINE_ID=az",
			"MYGO_WORKER_ID=az-worker-"+side,
			"AZ_DEVICE=cpu",
			"AZ_SIMULATIONS=8",
			"AZ_BATCH_SIZE=8",
		)
		cmd.Stdout = output
		cmd.Stderr = output
		if err := cmd.Start(); err != nil {
			t.Skipf("无法启动 AlphaZero worker: %v", err)
		}
		t.Cleanup(func() {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
		})
	}

	deadline := time.Now().Add(300 * time.Second)
	for time.Now().Before(deadline) {
		status, data := h.do("GET", "/api/v1/games/game_001", nil, nil)
		if status != 200 {
			t.Fatalf("查询棋局失败: %d %s", status, data)
		}
		game := decode(t, data)
		if game["status"] == "finished" {
			for _, output := range outputs {
				if !strings.Contains(output.String(), "落子成功") {
					t.Fatalf("两个 Worker 都必须落过子:\n%s", output.String())
				}
			}
			t.Logf("棋局结束，手数=%v 结果=%v", game["move_count"], game["result"])
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("AlphaZero worker 未能在时限内完成整局:\n%s\n%s", outputs[0].String(), outputs[1].String())
}
