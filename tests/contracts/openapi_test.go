package contracts

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestOpenAPIFreezesTheInterface(t *testing.T) {
	path := filepath.Join("..", "..", "api", "openapi.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("读取契约失败: %v", err)
	}
	contract := string(data)

	operations := regexp.MustCompile(`(?m)^\s{4}(get|post|patch|delete):\s*$`).FindAllStringSubmatch(contract, -1)
	found := make([]string, 0, len(operations))
	for _, match := range operations {
		found = append(found, match[1])
	}
	expected := []string{"post", "get", "get", "patch", "delete", "get", "post", "get", "post", "patch", "delete"}
	if len(found) != len(expected) {
		t.Fatalf("契约必须包含 %d 个操作, 实际 %d", len(expected), len(found))
	}
	for i, method := range expected {
		if found[i] != method {
			t.Fatalf("第 %d 个操作必须是 %s, 实际 %s", i+1, method, found[i])
		}
	}

	if !strings.Contains(contract, "url: http://127.0.0.1:8080/api/v1") {
		t.Fatalf("契约必须声明 /api/v1 前缀")
	}
	for _, path := range []string{
		"/games:",
		"/games/{id}:",
		"/games/{id}/moves:",
		"/engines:",
		"/ai/turn-claims:",
		"/ai/turn-claims/{id}:",
	} {
		if !strings.Contains(contract, "\n  "+path) {
			t.Fatalf("契约缺少路径 %s", path)
		}
	}

	for _, code := range []string{
		"INVALID_REQUEST", "UNAUTHORIZED", "FORBIDDEN", "NOT_YOUR_TURN", "GAME_NOT_FOUND",
		"CLAIM_NOT_FOUND", "VERSION_CONFLICT", "POSITION_OCCUPIED", "GAME_FINISHED",
		"GAME_PAUSED", "CLAIM_EXPIRED", "CLAIM_INVALID", "INVALID_GAME_CONFIG",
		"POSITION_OUT_OF_RANGE", "INTERNAL_ERROR",
	} {
		if !strings.Contains(contract, code) {
			t.Fatalf("契约缺少错误码 %s", code)
		}
	}
}
