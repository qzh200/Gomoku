package contracts

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// boardLine matches one row of the board the worker sends to the model.
var boardLine = regexp.MustCompile(`(?m)^(\d{2}) ((?:[.XO] ){14}[.XO])$`)

// stubOpenAI starts a server that speaks the OpenAI chat completions
// protocol. It reads the board out of the prompt and answers with a legal
// move, so the worker can play a whole game against the real API.
func stubOpenAI(t *testing.T) string {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		prompt := ""
		for _, message := range payload.Messages {
			if boardLine.MatchString(message.Content) {
				prompt = message.Content
			}
		}
		if prompt == "" {
			http.Error(w, "没有棋盘", http.StatusBadRequest)
			return
		}
		row, col, ok := stubMove(prompt)
		if !ok {
			http.Error(w, "读不出棋盘", http.StatusBadRequest)
			return
		}
		answer := fmt.Sprintf(`{"choices":[{"message":{"content":"{\"row\":%d,\"col\":%d}"}}]}`, row, col)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(answer))
	})
	return httptestServer(t, mux)
}

// stubMove reads the board of the prompt and returns the move the stub
// engine plays: it wins at once when it can, otherwise it extends its own
// longest line.
func stubMove(prompt string) (int, int, bool) {
	mark := byte('X')
	if strings.Contains(prompt, "当前执子方：白棋") {
		mark = 'O'
	}
	grid := [15][15]byte{}
	for _, match := range boardLine.FindAllStringSubmatch(prompt, -1) {
		row := int(match[1][0]-'0')*10 + int(match[1][1]-'0') - 1
		cells := strings.Split(strings.TrimSpace(match[2]), " ")
		for col, cell := range cells {
			grid[row][col] = cell[0]
		}
	}
	if row, col, ok := winningCell(grid, mark); ok {
		return row, col, true
	}
	if row, col, ok := extendCell(grid, mark); ok {
		return row, col, true
	}
	for row := range 15 {
		for col := range 15 {
			if grid[row][col] == '.' {
				return row, col, true
			}
		}
	}
	return 0, 0, false
}

// winningCell returns the cell that makes five in a row for the mark.
func winningCell(grid [15][15]byte, mark byte) (int, int, bool) {
	directions := [][2]int{{0, 1}, {1, 0}, {1, 1}, {1, -1}}
	for row := range 15 {
		for col := range 15 {
			if grid[row][col] != '.' {
				continue
			}
			for _, d := range directions {
				count := 1
				for sign := -1; sign <= 1; sign += 2 {
					r, c := row+d[0]*sign, col+d[1]*sign
					for r >= 0 && r < 15 && c >= 0 && c < 15 && grid[r][c] == mark {
						count++
						r += d[0] * sign
						c += d[1] * sign
					}
				}
				if count >= 5 {
					return row, col, true
				}
			}
		}
	}
	return 0, 0, false
}

// extendCell returns the cell that extends the longest own line.
func extendCell(grid [15][15]byte, mark byte) (int, int, bool) {
	directions := [][2]int{{0, 1}, {1, 0}, {1, 1}, {1, -1}}
	bestLength := 0
	bestRow, bestCol := -1, -1
	for row := range 15 {
		for col := range 15 {
			if grid[row][col] != mark {
				continue
			}
			for _, d := range directions {
				length := 1
				r, c := row+d[0], col+d[1]
				for r >= 0 && r < 15 && c >= 0 && c < 15 && grid[r][c] == mark {
					length++
					r += d[0]
					c += d[1]
				}
				if length < bestLength {
					continue
				}
				if r >= 0 && r < 15 && c >= 0 && c < 15 && grid[r][c] == '.' {
					bestLength = length
					bestRow, bestCol = r, c
				}
			}
		}
	}
	return bestRow, bestCol, bestRow >= 0
}

// TestLLMWorkerFinishesAGame runs the Python LLM worker against the real
// API and a stub model service, and checks that the game reaches its end.
func TestLLMWorkerFinishesAGame(t *testing.T) {
	python, err := exec.LookPath("python")
	if err != nil {
		t.Skip("找不到 python，跳过 LLM worker 测试")
	}
	root := filepath.Join("..", "..")
	if _, err := os.Stat(filepath.Join(root, "llm_worker", "__main__.py")); err != nil {
		t.Skip("找不到 llm_worker 包，跳过 LLM worker 测试")
	}
	h := newHarness(t)
	h.createGame("ai_vs_ai", aiVsAIPlayers())
	model := stubOpenAI(t)

	// The classical engine plays the other side, so the game can end.
	classical := exec.Command(python, filepath.Join("..", "..", "mock_worker", "mock_worker.py"),
		"--url", h.server.URL+"/api/v1",
		"--engine", "classical",
		"--token", engineTokens["classical"],
		"--worker-id", "classical-test-worker",
	)
	classicalOut := &strings.Builder{}
	classical.Stdout = classicalOut
	classical.Stderr = classicalOut
	if err := classical.Start(); err != nil {
		t.Skipf("无法启动 mock worker: %v", err)
	}
	t.Cleanup(func() {
		_ = classical.Process.Kill()
		_, _ = classical.Process.Wait()
	})

	cmd := exec.Command(python, "-m", "llm_worker")
	cmd.Dir = root
	cmd.Env = append(os.Environ(),
		"MYGO_URL="+h.server.URL+"/api/v1",
		"MYGO_ENGINE_TOKEN="+engineTokens["llm"],
		"MYGO_WORKER_ID=llm-test-worker",
		"OPENAI_BASE_URL="+model+"/v1",
		"OPENAI_API_KEY=test-key",
		"OPENAI_MODEL=stub-model",
	)
	output := &strings.Builder{}
	cmd.Stdout = output
	cmd.Stderr = output
	if err := cmd.Start(); err != nil {
		t.Skipf("无法启动 LLM worker: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})

	deadline := time.Now().Add(120 * time.Second)
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
				t.Fatalf("LLM worker 必须真的下过棋，手数=%v:\n%s", game["move_count"], output.String())
			}
			if !strings.Contains(output.String(), "落子成功") {
				t.Fatalf("LLM worker 必须至少成功落子一次:\n%s", output.String())
			}
			t.Logf("棋局结束，手数=%v 结果=%v", game["move_count"], game["result"])
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("LLM worker 未能在时限内完成整局:\n%s\n%s", output.String(), classicalOut.String())
}
