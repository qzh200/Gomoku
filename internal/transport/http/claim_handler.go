package http

import (
	"encoding/json"
	"net/http"
	"time"

	"mygo/internal/application"
	"mygo/internal/auth"
)

// submitMoveRequest is the wire format of the unified move request.
type submitMoveRequest struct {
	Row             int    `json:"row"`
	Col             int    `json:"col"`
	ExpectedVersion int    `json:"expected_version"`
	ClaimID         string `json:"claim_id"`
}

// engineResponse is one engine of the engine registry.
type engineResponse struct {
	EngineID     string     `json:"engine_id"`
	DisplayName  string     `json:"display_name"`
	Available    bool       `json:"available"`
	LastActiveAt *time.Time `json:"last_active_at"`
}

// engineListResponse is the wire format of the engine registry.
type engineListResponse struct {
	Items []engineResponse `json:"items"`
}

// claimRequest is the wire format of the turn claim request.
type claimRequest struct {
	WorkerID    string `json:"worker_id"`
	WaitSeconds int    `json:"wait_seconds"`
}

// renewRequest is the wire format of the lease renewal request.
type renewRequest struct {
	ExtendSeconds int `json:"extend_seconds"`
}

// handleSubmitMove applies a move of a human or an AI.
func (s *Server) handleSubmitMove(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	caller, ok := s.callerFor(w, r)
	if !ok {
		return
	}
	key := r.Header.Get("Idempotency-Key")
	if key == "" {
		writeError(w, &application.Error{Code: application.CodeInvalidRequest, Message: "缺少 Idempotency-Key 请求头"})
		return
	}
	var body submitMoveRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, &application.Error{Code: application.CodeInvalidRequest, Message: "请求体不是合法 JSON"})
		return
	}
	result, err := s.moves.Submit(id, caller, application.SubmitMoveRequest{
		Row:             body.Row,
		Col:             body.Col,
		ExpectedVersion: body.ExpectedVersion,
		ClaimID:         body.ClaimID,
		IdempotencyKey:  key,
	})
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, result)
}

// handleListEngines returns the engine registry.
func (s *Server) handleListEngines(w http.ResponseWriter, r *http.Request) {
	engines := s.auth.Engines()
	items := make([]engineResponse, 0, len(engines))
	for _, engine := range engines {
		items = append(items, engineResponse{
			EngineID:     engine.EngineID,
			DisplayName:  engine.DisplayName,
			Available:    engine.Available,
			LastActiveAt: engine.LastActiveAt,
		})
	}
	writeJSON(w, http.StatusOK, engineListResponse{Items: items})
}

// handleClaimTurn long polls for a turn of the calling engine.
func (s *Server) handleClaimTurn(w http.ResponseWriter, r *http.Request) {
	engine, ok := s.engineCaller(w, r)
	if !ok {
		return
	}
	var body claimRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, &application.Error{Code: application.CodeInvalidRequest, Message: "请求体不是合法 JSON"})
		return
	}
	if body.WorkerID == "" {
		writeError(w, &application.Error{Code: application.CodeInvalidRequest, Message: "缺少 worker_id"})
		return
	}
	if body.WaitSeconds < 0 || body.WaitSeconds > 120 {
		writeError(w, &application.Error{Code: application.CodeInvalidRequest, Message: "wait_seconds 超出范围"})
		return
	}
	s.auth.Touch(engine.EngineID)

	claim, granted, err := s.claims.Claim(r.Context(), engine.EngineID, body.WorkerID, time.Duration(body.WaitSeconds)*time.Second)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		writeError(w, err)
		return
	}
	if !granted {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	writeJSON(w, http.StatusCreated, claim)
}

// handleRenewClaim extends the lease of a claim.
func (s *Server) handleRenewClaim(w http.ResponseWriter, r *http.Request) {
	engine, ok := s.engineCaller(w, r)
	if !ok {
		return
	}
	var body renewRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, &application.Error{Code: application.CodeInvalidRequest, Message: "请求体不是合法 JSON"})
		return
	}
	lease, err := s.claims.Renew(r.PathValue("id"), engine.EngineID, body.ExtendSeconds)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"claim_id":   lease.ClaimID,
		"expires_at": lease.ExpiresAt.UTC(),
	})
}

// handleReleaseClaim drops the lease of a claim.
func (s *Server) handleReleaseClaim(w http.ResponseWriter, r *http.Request) {
	engine, ok := s.engineCaller(w, r)
	if !ok {
		return
	}
	if err := s.claims.Release(r.PathValue("id"), engine.EngineID); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// callerFor resolves the caller of a move request. A bearer token identifies
// an AI engine, an owner token identifies the owner of a game.
func (s *Server) callerFor(w http.ResponseWriter, r *http.Request) (application.Caller, bool) {
	if token := auth.BearerToken(r.Header.Get("Authorization")); token != "" {
		engine, ok := s.auth.EngineByToken(token)
		if !ok {
			writeError(w, &application.Error{Code: application.CodeUnauthorized, Message: "引擎凭据无效"})
			return application.Caller{}, false
		}
		s.auth.Touch(engine.EngineID)
		return application.Caller{Kind: application.CallerEngine, EngineID: engine.EngineID}, true
	}
	if token := r.Header.Get("X-Game-Owner-Token"); token != "" {
		return application.Caller{Kind: application.CallerOwner, OwnerTokenHash: auth.HashToken(token)}, true
	}
	writeError(w, &application.Error{Code: application.CodeUnauthorized, Message: "缺少有效凭据"})
	return application.Caller{}, false
}

// engineCaller resolves the engine of a request. The engine comes from the
// credential, never from a request parameter.
func (s *Server) engineCaller(w http.ResponseWriter, r *http.Request) (*auth.Engine, bool) {
	token := auth.BearerToken(r.Header.Get("Authorization"))
	if token == "" {
		writeError(w, &application.Error{Code: application.CodeUnauthorized, Message: "缺少引擎凭据"})
		return nil, false
	}
	engine, ok := s.auth.EngineByToken(token)
	if !ok {
		writeError(w, &application.Error{Code: application.CodeUnauthorized, Message: "引擎凭据无效"})
		return nil, false
	}
	if !engine.Available {
		writeError(w, &application.Error{Code: application.CodeForbidden, Message: "引擎当前不可用"})
		return nil, false
	}
	return engine, true
}
