package http

import (
	"net/http"

	"mygo/internal/application"
	"mygo/internal/auth"
)

type Server struct {
	games  *application.GameService
	moves  *application.MoveService
	claims *application.ClaimService
	auth   *auth.Registry
}

func NewServer(games *application.GameService, moves *application.MoveService, claims *application.ClaimService, registry *auth.Registry) *Server {
	return &Server{games: games, moves: moves, claims: claims, auth: registry}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/games", s.handleCreateGame)
	mux.HandleFunc("GET /api/v1/games", s.handleListGames)
	mux.HandleFunc("GET /api/v1/games/{id}", s.handleGetGame)
	mux.HandleFunc("PATCH /api/v1/games/{id}", s.handlePatchGame)
	mux.HandleFunc("DELETE /api/v1/games/{id}", s.handleDeleteGame)
	mux.HandleFunc("GET /api/v1/games/{id}/moves", s.handleListMoves)
	mux.HandleFunc("POST /api/v1/games/{id}/moves", s.handleSubmitMove)
	mux.HandleFunc("GET /api/v1/engines", s.handleListEngines)
	mux.HandleFunc("POST /api/v1/ai/turn-claims", s.handleClaimTurn)
	mux.HandleFunc("PATCH /api/v1/ai/turn-claims/{id}", s.handleRenewClaim)
	mux.HandleFunc("DELETE /api/v1/ai/turn-claims/{id}", s.handleReleaseClaim)
	return mux
}
