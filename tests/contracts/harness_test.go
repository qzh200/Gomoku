// Package contracts tests the frozen RESTful API of MyGo over HTTP.
package contracts

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"mygo/internal/application"
	"mygo/internal/auth"
	"mygo/internal/repository"
	"mygo/internal/scheduler"
	httpapi "mygo/internal/transport/http"
)

// engineTokens are the bearer tokens of the preset engines in the tests.
var engineTokens = map[string]string{
	"classical": "classical-test-token",
	"llm":       "llm-test-token",
	"az":        "az-test-token",
}

// harness is a running MyGo API with its own storage.
type harness struct {
	t      *testing.T
	server *httptest.Server
	local  string
	keys   atomic.Uint64
}

// newHarness starts an API with the preset engines and fixed tokens.
func newHarness(t *testing.T) *harness {
	return newHarnessWithLease(t, 0)
}

// newHarnessWithLease starts an API whose leases live for the given time.
// A lease time of zero uses the default.
func newHarnessWithLease(t *testing.T, leaseTTL time.Duration) *harness {
	t.Helper()
	return newHarnessWithEngines(t, leaseTTL, "classical", "llm")
}

// newHarnessWithEngines starts an API with the given engines and fixed tokens.
func newHarnessWithEngines(t *testing.T, leaseTTL time.Duration, engineIDs ...string) *harness {
	t.Helper()
	engines := make([]auth.EngineConfig, 0, len(engineIDs))
	for _, engineID := range engineIDs {
		token, ok := engineTokens[engineID]
		if !ok {
			t.Fatalf("测试没有引擎 %q 的令牌", engineID)
		}
		engines = append(engines, auth.EngineConfig{
			EngineID:    engineID,
			DisplayName: engineID,
			Token:       token,
		})
	}
	cfg := auth.Config{
		LocalClientToken: "local-test-token",
		Engines:          engines,
	}
	registry, err := auth.NewRegistry(cfg)
	if err != nil {
		t.Fatalf("构建引擎注册表失败: %v", err)
	}
	coord := application.NewCoordinator(repository.NewMemoryGames(), scheduler.NewLeaseManager(), repository.NewMemoryIdempotency())
	server := httpapi.NewServer(
		application.NewGameService(coord, registry),
		application.NewMoveService(coord),
		application.NewClaimService(coord, application.LeaseTTL(leaseTTL)),
		registry,
	)
	h := &harness{t: t, server: httptest.NewServer(server.Handler()), local: "local-test-token"}
	t.Cleanup(h.server.Close)
	return h
}

// do sends a request and returns the status and the body.
func (h *harness) do(method, path string, body any, headers map[string]string) (int, []byte) {
	h.t.Helper()
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			h.t.Fatalf("序列化请求体失败: %v", err)
		}
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, h.server.URL+path, reader)
	if err != nil {
		h.t.Fatalf("构建请求失败: %v", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for name, value := range headers {
		req.Header.Set(name, value)
	}
	if method == http.MethodPost && strings.HasSuffix(path, "/moves") {
		if req.Header.Get("Idempotency-Key") == "" {
			req.Header.Set("Idempotency-Key", "key-"+strconv.FormatUint(h.keys.Add(1), 10))
		}
	}
	resp, err := h.server.Client().Do(req)
	if err != nil {
		h.t.Fatalf("请求失败: %v", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		h.t.Fatalf("读取响应失败: %v", err)
	}
	return resp.StatusCode, data
}

// localHeaders returns the headers of the local interface.
func (h *harness) localHeaders() map[string]string {
	return map[string]string{"X-Local-Client-Token": h.local}
}

// ownerHeaders returns the headers of the owner of a game.
func ownerHeaders(token string) map[string]string {
	return map[string]string{"X-Game-Owner-Token": token}
}

// engineHeaders returns the headers of an AI engine.
func engineHeaders(engineID string) map[string]string {
	return map[string]string{"Authorization": "Bearer " + engineTokens[engineID]}
}

// httptestServer starts a server for a test and returns its URL.
func httptestServer(t *testing.T, handler http.Handler) string {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server.URL
}

// decode unmarshals a response body.
func decode(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("解析响应失败: %v (%s)", err, data)
	}
	return out
}

// errorCode returns the error code of an error response.
func errorCode(t *testing.T, data []byte) string {
	t.Helper()
	out := decode(t, data)
	errObj, ok := out["error"].(map[string]any)
	if !ok {
		t.Fatalf("响应缺少 error 字段: %s", data)
	}
	code, _ := errObj["code"].(string)
	return code
}
