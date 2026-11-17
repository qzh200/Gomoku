package application

import (
	"testing"

	"mygo/internal/auth"
	"mygo/internal/domain/gomoku"
	"mygo/internal/repository"
	"mygo/internal/scheduler"
)

func engineSide(id string) gomoku.PlayerConfig {
	return gomoku.PlayerConfig{Type: gomoku.TypeAI, EngineID: &id}
}

func TestUndoCountTakesBackTheWholeExchange(t *testing.T) {
	human := gomoku.PlayerConfig{Type: gomoku.TypeHuman}
	game := gomoku.NewGame("g", gomoku.ModeHumanVsAI, engineSide("ai"), human)

	game.Place(7, 7, gomoku.PlayerBlack)
	game.Place(7, 8, gomoku.PlayerWhite)
	if got := undoCount(game); got != 1 {
		t.Fatalf("AI 落子后人类刚落子, 只撤销人类这一步: 实际 %d", got)
	}

	game.Place(8, 8, gomoku.PlayerBlack)
	if got := undoCount(game); got != 2 {
		t.Fatalf("AI 已经应手, 悔棋必须一并撤销双方这两步: 实际 %d", got)
	}

	game.Place(6, 8, gomoku.PlayerWhite)
	if got := undoCount(game); got != 1 {
		t.Fatalf("轮人类落子时只撤销人类这一步: 实际 %d", got)
	}
}

func TestUndoThroughServiceRestoresThePosition(t *testing.T) {
	games := repository.NewMemoryGames()
	coord := NewCoordinator(games, scheduler.NewLeaseManager(), repository.NewMemoryIdempotency())
	service := NewMoveService(coord)

	human := gomoku.PlayerConfig{Type: gomoku.TypeHuman}
	game := gomoku.NewGame("g1", gomoku.ModeHumanVsAI, human, engineSide("ai"))
	token := "owner"
	if err := games.Create(game, auth.HashToken(token)); err != nil {
		t.Fatalf("创建棋局失败: %v", err)
	}
	if err := games.Update("g1", func(stored *gomoku.Game) error {
		stored.Place(7, 7, gomoku.PlayerBlack)
		stored.Place(7, 8, gomoku.PlayerWhite)
		return nil
	}); err != nil {
		t.Fatalf("落子失败: %v", err)
	}

	caller := Caller{Kind: CallerOwner, OwnerTokenHash: auth.HashToken(token)}
	updated, err := service.Undo("g1", caller, UndoRequest{})
	if err != nil {
		t.Fatalf("悔棋失败: %v", err)
	}
	if updated.MoveCount != 0 || updated.Board[7][7] != 0 || updated.Board[7][8] != 0 {
		t.Fatalf("悔棋必须撤销双方最后两步, 实际手数 %d", updated.MoveCount)
	}
	if updated.Current != gomoku.PlayerBlack {
		t.Fatalf("悔棋后必须轮到黑棋")
	}
	if updated.Version <= 2 {
		t.Fatalf("悔棋必须推进版本号, 实际 %d", updated.Version)
	}
}
