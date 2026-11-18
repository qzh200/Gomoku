// Command launcher 以解压目录为根，启动 Go 主程序与 Python Worker。
//
// 布局（由 packaging/build.ps1 生成）：
//
//	Gomoku.exe        本启动器
//	mygo.exe          Go 主程序（API + 桌面 UI）；注意不能与启动器同名（NTFS 不区分大小写）
//	configs/          引擎注册表
//	models/az/        AlphaZero 权重
//	workers/          冻结的 Python Worker 运行时（torch/numpy 已在内）
//	classical_worker/ 每个 Worker 一个目录，放 .env 参数文件
//	llm_worker/       └ LLM 引擎
//	az_worker/        └ AlphaZero 引擎
//
// 没有 workers/workers.exe 时退回到源码模式：用 .venv 或 PATH 上的 Python 跑
// packaging/worker_entry.py。直接运行 mygo.exe 仍然可用，但不会拉起 Worker。
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// entryPoint 是源码模式下的 Worker 派发脚本（与 PyInstaller 入口同一份）。
const entryPoint = "packaging/worker_entry.py"

const (
	serveReady    = 15 * time.Second
	serveInterval = 200 * time.Millisecond
	stopGrace     = 3 * time.Second
)

// worker 描述一个随主程序启动的 Python Worker。
type worker struct {
	name    string   // 日志名，同时作为 --worker 的参数
	modules []string // 该 Worker 需要的第三方模块
	envFile string   // 相对根目录的参数文件
}

var workers = []worker{
	{name: "classical", envFile: "classical_worker/.env"},
	{name: "llm", envFile: "llm_worker/.env"},
	{name: "az", modules: []string{"torch", "numpy"}, envFile: "az_worker/.env"},
}

func main() {
	root, err := executableDir()
	if err != nil {
		fatal("定位程序目录失败: %v", err)
	}
	app := filepath.Join(root, "mygo.exe")
	if _, err := os.Stat(app); err != nil {
		fatal("找不到 %s，请完整解压发布包再运行", app)
	}
	// 同名会无限自我复制：Windows 路径不分大小写，曾把启动器写成 MyGo.exe
	// 覆盖掉 mygo.exe，结果每次启动都把自己当主程序拉起。
	if self, err := os.Executable(); err == nil && sameFile(self, app) {
		fatal("启动器与主程序是同一个文件（%s），请重新解压发行包", app)
	}

	// 主程序与 Worker 都按相对路径找 configs/ 与 models/，因此切到根目录。
	if err := os.Chdir(root); err != nil {
		fatal("切换到 %s 失败: %v", root, err)
	}

	started := time.Now()
	parent := exec.Command(app)
	parent.Dir = root
	parent.Stdout = os.Stdout
	parent.Stderr = os.Stderr
	if err := parent.Start(); err != nil {
		fatal("启动 mygo.exe 失败: %v", err)
	}
	if err := waitRuntime(root, started); err != nil {
		fmt.Fprintln(os.Stderr, "启动器:", err)
		_ = parent.Process.Kill()
		return
	}

	// 发行包内用打包好的 workers.exe（依赖已冻结），否则用源码 + Python。
	runtime, isFrozen := workerRuntime(root)
	var children []*exec.Cmd
	if runtime == "" {
		fmt.Fprintln(os.Stderr, "没有可用的 Worker 运行时：既无 workers.exe，也无 Python")
	} else {
		for _, w := range workers {
			if !isFrozen && w.modules != nil && !hasModules(runtime, w.modules) {
				fmt.Fprintf(os.Stderr, "跳过 %s Worker：当前 Python 缺少 %v（%s）\n", w.name, w.modules, runtime)
				continue
			}
			if cmd := startWorker(root, runtime, isFrozen, w); cmd != nil {
				children = append(children, cmd)
			}
		}
	}

	err = parent.Wait()
	for _, cmd := range children {
		terminate(cmd)
	}
	flushAll()
	if err != nil {
		// 用户关窗口时退出码非 0 属正常，不再放大。
		fmt.Fprintln(os.Stderr, "mygo 退出:", err)
	}
}

// waitRuntime 等主程序写出 engines.runtime.json。
// Worker 从该文件读自己的令牌，主程序每次启动都会重新生成，所以顺序不能反；
// 只认 started 之后写入的文件，否则上一轮遗留的副本会让 Worker 拿到过期令牌。
func waitRuntime(root string, started time.Time) error {
	path := filepath.Join(root, "configs", "engines.runtime.json")
	deadline := time.Now().Add(serveReady)
	for time.Now().Before(deadline) {
		if info, err := os.Stat(path); err == nil && info.ModTime().After(started) {
			return nil
		}
		time.Sleep(serveInterval)
	}
	return fmt.Errorf("等待 %s 超时，Worker 不会启动（令牌缺失）", path)
}

// workerRuntime 返回 Worker 解释器：发行包自带的 workers.exe，其次工程 venv，
// 最后 PATH 上的 python。第二个返回值为 true 表示已冻结全部依赖。
func workerRuntime(root string) (string, bool) {
	if runtime.GOOS == "windows" {
		exe := filepath.Join(root, "workers", "workers.exe")
		if _, err := os.Stat(exe); err == nil {
			return exe, true
		}
	}
	for _, candidate := range []string{
		filepath.Join(root, ".venv", "Scripts", "python.exe"),
		filepath.Join(root, ".venv", "bin", "python"),
	} {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, false
		}
	}
	if path, err := exec.LookPath("python"); err == nil {
		return path, false
	}
	return "", false
}

// hasModules 用一次子进程确认依赖齐全；缺依赖的 Worker 不启动，避免刷屏重连。
func hasModules(python string, modules []string) bool {
	args := []string{"-c", "import sys"}
	for _, name := range modules {
		args[1] += ";import " + name
	}
	cmd := exec.Command(python, args...)
	cmd.Env = append(os.Environ(), "PYTHONPATH=.")
	return cmd.Run() == nil
}

func startWorker(root, runtime string, isFrozen bool, w worker) *exec.Cmd {
	envPath := filepath.Join(root, w.envFile)
	if _, err := os.Stat(envPath); err != nil {
		fmt.Fprintf(os.Stderr, "跳过 %s Worker：缺少参数文件 %s\n", w.name, envPath)
		return nil
	}
	var cmd *exec.Cmd
	if isFrozen {
		cmd = exec.Command(runtime, "--worker", w.name, "--env-file", envPath)
	} else {
		cmd = exec.Command(runtime, entryPoint, "--worker", w.name, "--env-file", envPath)
	}
	cmd.Dir = root
	cmd.Env = append(os.Environ(),
		"PYTHONPATH=.",
		"PYTHONUTF8=1",
		"PYTHONUNBUFFERED=1",
		"MYGO_RUNTIME_CONFIG=configs/engines.runtime.json",
	)
	cmd.Stdout = prefixed(w.name)
	cmd.Stderr = prefixed(w.name)
	hideWindow(cmd)
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "启动 %s Worker 失败: %v\n", w.name, err)
		return nil
	}
	return cmd
}

// sameFile 比较真实路径，绕开大小写与短文件名差异。
func sameFile(a, b string) bool {
	left, right := filepath.Clean(a), filepath.Clean(b)
	if strings.EqualFold(left, right) {
		return true
	}
	absA, errA := filepath.Abs(left)
	absB, errB := filepath.Abs(right)
	return errA == nil && errB == nil && strings.EqualFold(absA, absB)
}

func terminate(cmd *exec.Cmd) {
	done := make(chan struct{})
	go func() {
		_ = cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
		return
	default:
	}
	_ = cmd.Process.Kill()
	select {
	case <-done:
	case <-time.After(stopGrace):
	}
}
