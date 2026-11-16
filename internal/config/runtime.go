package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"mygo/internal/auth"
)

// RuntimePath is the file the effective engine registry is written to. A
// worker reads it to learn the bearer token of its engine, so the token
// never has to be copied by hand.
const RuntimePath = "configs/engines.runtime.json"

// RuntimePathEnv names the environment variable that overrides the path of
// the runtime registry.
const RuntimePathEnv = "MYGO_RUNTIME_CONFIG"

// SaveRuntime writes the effective engine registry, tokens included, so that
// a worker can find the token of its engine.
//
// The file holds a credential, so it is written for the owner only. It is
// never served over HTTP and never shown in a game.
func SaveRuntime(path string, cfg auth.Config) error {
	if path == "" {
		return fmt.Errorf("运行时配置路径为空")
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("序列化运行时配置失败: %w", err)
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return fmt.Errorf("创建目录 %s 失败: %w", dir, err)
		}
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("写入运行时配置 %s 失败: %w", path, err)
	}
	return nil
}
