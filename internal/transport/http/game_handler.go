package http

import (
	"encoding/json"
	"net/http"

	"mygo/internal/application"
	"mygo/internal/auth"
	"mygo/internal/domain/gomoku"
)

// createGameRequest is the wire format of the create game request.
type createGameRequest struct {
	Mode    gomoku.Mode `json:"mode"`
	Players map[string]struct {
		Type     gomoku.PlayerType `json:"type"`
		EngineID string            `json:"engine_id"`
		Config   map[string]any    `json:"config"`
	} `json:"players"`
}

// patchGameRequest is the wire format of the pause and resume request.
type patchGameRequest struct {
	ExecutionState  gomoku.ExecutionState `json:"execution_state"`
	ExpectedVersion *int                  `json:"expected_version"`
}

// moveListResponse is the wire format of the move history.
type moveListResponse struct {
	Items []gomoku.Move `json:"items"`
}

// gameListResponse is the wire format of the game list.
type gameListResponse struct {
	Items []*gomoku.Game `json:"items"`
}

// handleCreateGame creates a game. The local client token authorizes it.
func (s *Server) handleCreateGame(w http.ResponseWriter, r *http.Request) {
	token := r.Header.Get("X-Local-Client-Token")
	if token == "" {
		writeError(w, &application.Error{Code: application.CodeUnauthorized, Message: "缺少本地客户端凭据"})
		return
	}
	if !s.auth.IsLocalClient(token) {
		writeError(w, &application.Error{Code: application.CodeUnauthorized, Message: "本地客户端凭据无效"})
		return
	}

	var body createGameRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, &application.Error{Code: application.CodeInvalidRequest, Message: "请求体不是合法 JSON"})
		return
	}
	players := make(map[string]application.CreatePlayer, len(body.Players))
	for side, player := range body.Players {
		players[side] = application.CreatePlayer{Type: player.Type, EngineID: player.EngineID, Config: player.Config}
	}
	result, err := s.games.Create(auth.HashToken(token), application.CreateGameRequest{Mode: body.Mode, Players: players}, r.Header.Get("Idempotency-Key"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, result)
}

// handleListGames returns the snapshots of all games. Reading is open: the
// service only listens on the loopback interface and never returns a
// credential.
func (s *Server) handleListGames(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, gameListResponse{Items: s.games.List()})
}

// handleGetGame returns the snapshot of one game.
func (s *Server) handleGetGame(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	game, err := s.games.Get(id)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, game)
}

// handlePatchGame pauses or resumes an automated game.
func (s *Server) handlePatchGame(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !s.authorizedOwner(w, r, id) {
		return
	}
	var body patchGameRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, &application.Error{Code: application.CodeInvalidRequest, Message: "请求体不是合法 JSON"})
		return
	}
	game, err := s.games.SetExecutionState(id, body.ExecutionState, body.ExpectedVersion)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, game)
}

// handleDeleteGame removes a game.
func (s *Server) handleDeleteGame(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !s.authorizedOwner(w, r, id) {
		return
	}
	if err := s.games.Delete(id); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleListMoves returns the move history of a game.
func (s *Server) handleListMoves(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	items, err := s.moves.History(id)
	if err != nil {
		writeError(w, err)
		return
	}
	if items == nil {
		items = []gomoku.Move{}
	}
	writeJSON(w, http.StatusOK, moveListResponse{Items: items})
}

// authorizedOwner checks that the request carries the owner token of the
// game. It writes the error response itself.
func (s *Server) authorizedOwner(w http.ResponseWriter, r *http.Request, id string) bool {
	token := r.Header.Get("X-Game-Owner-Token")
	if token == "" {
		writeError(w, &application.Error{Code: application.CodeUnauthorized, Message: "缺少棋局 Owner 凭据"})
		return false
	}
	caller := application.Caller{Kind: application.CallerOwner, OwnerTokenHash: auth.HashToken(token)}
	if _, err := s.moves.CheckOwner(id, caller); err != nil {
		writeError(w, err)
		return false
	}
	return true
}
