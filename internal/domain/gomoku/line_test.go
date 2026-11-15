package gomoku

import "testing"

func TestWinningLineFindsTheFive(t *testing.T) {
	var board Board
	for col := 5; col <= 9; col++ {
		board.Place(7, col, PlayerBlack)
	}
	line := WinningLine(board, PlayerBlack)
	if len(line) != WinLength {
		t.Fatalf("必须找到 %d 颗连成一线, 实际 %d", WinLength, len(line))
	}
	if line[0] != [2]int{7, 5} || line[4] != [2]int{7, 9} {
		t.Fatalf("连线必须按顺序返回: %v", line)
	}
	if WinningLine(board, PlayerWhite) != nil {
		t.Fatalf("白棋没有连线")
	}
}

func TestWinningLineOnDiagonal(t *testing.T) {
	var board Board
	for i := range 5 {
		board.Place(3+i, 3+i, PlayerWhite)
	}
	line := WinningLine(board, PlayerWhite)
	if len(line) != WinLength {
		t.Fatalf("必须找到对角线上的五子, 实际 %d", len(line))
	}
	if line[0] != [2]int{3, 3} || line[4] != [2]int{7, 7} {
		t.Fatalf("连线必须按顺序返回: %v", line)
	}
}

func TestWinningLineNeedsFive(t *testing.T) {
	var board Board
	for col := 5; col <= 8; col++ {
		board.Place(7, col, PlayerBlack)
	}
	if WinningLine(board, PlayerBlack) != nil {
		t.Fatalf("四子不成线")
	}
}
