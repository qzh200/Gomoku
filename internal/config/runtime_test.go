package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"mygo/internal/auth"
)

func TestSaveRuntimeWritesTheTokens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "engines.runtime.json")
	available := true
	err := SaveRuntime(path, auth.Config{
		Engines: []auth.EngineConfig{
			{EngineID: "classical", DisplayName: "Classical AI", Token: "c-token", Available: &available},
			{EngineID: "llm", DisplayName: "LLM AI", Token: "l-token", Available: &available},
		},
		LocalClientToken: "local-token",
	})
	if err != nil {
		t.Fatalf("写入运行时配置失败: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("读取运行时配置失败: %v", err)
	}
	var cfg auth.Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("解析运行时配置失败: %v", err)
	}
	if cfg.LocalClientToken != "local-token" {
		t.Fatalf("运行时配置必须带本地客户端令牌: %q", cfg.LocalClientToken)
	}
	if len(cfg.Engines) != 2 {
		t.Fatalf("运行时配置必须有两个引擎: %v", cfg.Engines)
	}
	if cfg.Engines[1].EngineID != "llm" || cfg.Engines[1].Token != "l-token" {
		t.Fatalf("运行时配置必须带引擎令牌: %v", cfg.Engines[1])
	}
}

func TestSaveRuntimeCreatesTheDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "engines.runtime.json")
	if err := SaveRuntime(path, auth.Config{}); err != nil {
		t.Fatalf("写入运行时配置失败: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("运行时配置必须写出来: %v", err)
	}
}

func TestSaveRuntimeRejectsAnEmptyPath(t *testing.T) {
	if err := SaveRuntime("", auth.Config{}); err == nil {
		t.Fatalf("空路径必须报错")
	}
}
