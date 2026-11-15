// Package config loads the engine registry configuration.
package config

import (
	"encoding/json"
	"fmt"
	"os"

	"mygo/internal/auth"
)

// Path is the default location of the configuration file.
const Path = "configs/engines.json"

// Load reads the engine configuration. A missing file is not an error: the
// process then starts with the preset engines and random tokens.
func Load(path string) (auth.Config, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return defaultConfig(), false, nil
		}
		return auth.Config{}, false, fmt.Errorf("读取配置 %s 失败: %w", path, err)
	}
	var cfg auth.Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return auth.Config{}, true, fmt.Errorf("解析配置 %s 失败: %w", path, err)
	}
	return cfg, true, nil
}

// defaultConfig returns the preset engines without tokens.
func defaultConfig() auth.Config {
	return auth.Config{Engines: []auth.EngineConfig{
		{EngineID: "classical", DisplayName: "Classical AI"},
		{EngineID: "llm", DisplayName: "LLM AI"},
	}}
}
