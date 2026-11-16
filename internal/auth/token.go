package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	ErrUnauthorized = errors.New("unauthorized")
	ErrForbidden    = errors.New("forbidden")
)

type Engine struct {
	EngineID string

	DisplayName string

	TokenHash string

	Token string

	Generated    bool
	Available    bool
	LastActiveAt *time.Time
}

type EngineConfig struct {
	EngineID    string `json:"engine_id"`
	DisplayName string `json:"display_name"`
	Token       string `json:"token"`
	Available   *bool  `json:"available"`
}

type Config struct {
	Engines          []EngineConfig `json:"engines"`
	LocalClientToken string         `json:"local_client_token"`
}

type Registry struct {
	mu         sync.RWMutex
	engines    map[string]*Engine
	byToken    map[string]*Engine
	localHash  string
	localToken string
}

func NewRegistry(cfg Config) (*Registry, error) {
	r := &Registry{
		engines: make(map[string]*Engine, len(cfg.Engines)),
		byToken: make(map[string]*Engine, len(cfg.Engines)),
	}
	for _, entry := range cfg.Engines {
		if entry.EngineID == "" {
			return nil, errors.New("engine without engine_id")
		}
		if _, ok := r.engines[entry.EngineID]; ok {
			return nil, fmt.Errorf("duplicate engine %q", entry.EngineID)
		}
		token := entry.Token
		generated := false
		if token == "" {
			token = RandomToken()
			generated = true
		}
		available := true
		if entry.Available != nil {
			available = *entry.Available
		}
		name := entry.DisplayName
		if name == "" {
			name = entry.EngineID
		}
		engine := &Engine{
			EngineID:    entry.EngineID,
			DisplayName: name,
			TokenHash:   HashToken(token),
			Token:       token,
			Generated:   generated,
			Available:   available,
		}
		r.engines[engine.EngineID] = engine
		r.byToken[engine.TokenHash] = engine
	}
	local := cfg.LocalClientToken
	if local == "" {
		local = RandomToken()
	}
	r.localHash = HashToken(local)
	r.localToken = local
	return r, nil
}

func (r *Registry) LocalToken() string { return r.localToken }

func (r *Registry) Engines() []*Engine {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Engine, 0, len(r.engines))
	for _, engine := range r.engines {
		copied := *engine
		out = append(out, &copied)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].EngineID < out[j].EngineID })
	return out
}

func (r *Registry) EngineByID(engineID string) (*Engine, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	engine, ok := r.engines[engineID]
	if !ok {
		return nil, false
	}
	copied := *engine
	return &copied, true
}

func (r *Registry) EngineByToken(token string) (*Engine, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	engine, ok := r.byToken[HashToken(token)]
	if !ok {
		return nil, false
	}
	copied := *engine
	return &copied, true
}

func (r *Registry) IsLocalClient(token string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.localHash == HashToken(token)
}

func (r *Registry) Touch(engineID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if engine, ok := r.engines[engineID]; ok {
		now := time.Now().UTC()
		engine.LastActiveAt = &now
	}
}

func HashToken(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

func RandomToken() string {
	buf := make([]byte, 24)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("token_%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(buf)
}

func BearerToken(header string) string {
	const prefix = "Bearer "
	if len(header) <= len(prefix) || !strings.EqualFold(header[:len(prefix)], prefix) {
		return ""
	}
	return strings.TrimSpace(header[len(prefix):])
}
