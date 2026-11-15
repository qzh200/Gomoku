package gomoku

import "testing"

func TestWinsInFourDirections(t *testing.T) {
	cases := []struct {
		name   string
		stones [][2]int
	}{
		{"horizontal", [][2]int{{7, 5}, {7, 6}, {7, 7}, {7, 8}, {7, 9}}},
		{"vertical", [][2]int{{5, 7}, {6, 7}, {7, 7}, {8, 7}, {9, 7}}},
		{"main diagonal", [][2]int{{5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}}},
		{"anti diagonal", [][2]int{{5, 9}, {6, 8}, {7, 7}, {8, 6}, {9, 5}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var board Board
			for _, stone := range tc.stones[:4] {
				board.Place(stone[0], stone[1], PlayerBlack)
			}
			if WinsAt(board, tc.stones[4][0], tc.stones[4][1], PlayerBlack) {
				t.Fatalf("four stones must not win")
			}
			board.Place(tc.stones[4][0], tc.stones[4][1], PlayerBlack)
			if !WinsAt(board, tc.stones[4][0], tc.stones[4][1], PlayerBlack) {
				t.Fatalf("five stones in a line must win")
			}
		})
	}
}

func TestSixInARowWins(t *testing.T) {
	var board Board
	for col := 3; col <= 8; col++ {
		board.Place(7, col, PlayerBlack)
	}
	if !WinsAt(board, 7, 8, PlayerBlack) {
		t.Fatalf("six in a row must win")
	}
}

func TestBlockedLineDoesNotWin(t *testing.T) {
	var board Board
	for _, col := range []int{5, 6, 8, 9} {
		board.Place(7, col, PlayerBlack)
	}
	board.Place(7, 7, PlayerWhite)
	if WinsAt(board, 7, 9, PlayerBlack) {
		t.Fatalf("a broken line must not win")
	}
}

func TestPlaceUpdatesState(t *testing.T) {
	game := NewGame("game_001", ModeHumanVsHuman,
		PlayerConfig{Type: TypeHuman, Config: map[string]any{}},
		PlayerConfig{Type: TypeHuman, Config: map[string]any{}})

	if !game.Place(7, 7, PlayerBlack) {
		t.Fatalf("first move must be accepted")
	}
	if game.Version != 1 || game.MoveCount != 1 {
		t.Fatalf("version and move count must advance: %d %d", game.Version, game.MoveCount)
	}
	if game.Current != PlayerWhite {
		t.Fatalf("the turn must pass to white")
	}
	if game.LastMove == nil || game.LastMove.Row != 7 || game.LastMove.Col != 7 || game.LastMove.Number != 1 {
		t.Fatalf("last move must be recorded")
	}
	if game.Place(7, 7, PlayerWhite) {
		t.Fatalf("an occupied cell must be rejected")
	}
	if game.Version != 1 {
		t.Fatalf("a rejected move must not change the version")
	}
}

func TestGameFinishesOnWin(t *testing.T) {
	game := NewGame("game_001", ModeHumanVsHuman,
		PlayerConfig{Type: TypeHuman, Config: map[string]any{}},
		PlayerConfig{Type: TypeHuman, Config: map[string]any{}})
	for col := 5; col <= 9; col++ {
		game.Place(7, col, PlayerBlack)
	}
	if game.Status != StatusFinished || game.Result != ResultBlackWin {
		t.Fatalf("the game must end with a black win: %s %s", game.Status, game.Result)
	}
	if game.Winner == nil || *game.Winner != PlayerBlack {
		t.Fatalf("the winner must be black")
	}
}

func TestDrawWhenBoardIsFull(t *testing.T) {
	game := NewGame("game_001", ModeHumanVsHuman,
		PlayerConfig{Type: TypeHuman, Config: map[string]any{}},
		PlayerConfig{Type: TypeHuman, Config: map[string]any{}})
	// A full board without a winning line ends the game as a draw. The cells
	// are filled directly because alternating turns always produce a line of
	// five before the board is full.
	for row := range BoardSize {
		for col := range BoardSize {
			player := Black
			if (row+col)%2 == 0 {
				player = White
			}
			game.Board[row][col] = player
		}
	}
	if !game.Board.IsFull() {
		t.Fatalf("the board must be full")
	}
	game.resolveDraw()
	if game.Status != StatusFinished || game.Result != ResultDraw {
		t.Fatalf("a full board must be a draw: %s %s", game.Status, game.Result)
	}
	if game.Winner != nil {
		t.Fatalf("a draw has no winner")
	}
}

func TestOutOfBoundsIsRejected(t *testing.T) {
	var board Board
	if board.Place(-1, 0, PlayerBlack) || board.Place(0, 15, PlayerBlack) {
		t.Fatalf("out of bounds moves must be rejected")
	}
	if !InBounds(0, 0) || !InBounds(14, 14) || InBounds(15, 0) || InBounds(0, -1) {
		t.Fatalf("bounds check is wrong")
	}
}
