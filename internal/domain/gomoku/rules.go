package gomoku

// WinLength is the number of consecutive stones a player needs to win.
const WinLength = 5

// directions are the four axes a winning line can follow.
var directions = [4][2]int{
	{0, 1},  // horizontal
	{1, 0},  // vertical
	{1, 1},  // main diagonal
	{1, -1}, // anti diagonal
}

// WinsAt reports whether the stone at row, col completes a line of at least
// WinLength stones of the same player.
func WinsAt(b Board, row, col int, player Player) bool {
	if !InBounds(row, col) || b[row][col] != int(player) {
		return false
	}
	for _, d := range directions {
		count := 1
		for _, sign := range [2]int{-1, 1} {
			r, c := row+d[0]*sign, col+d[1]*sign
			for InBounds(r, c) && b[r][c] == int(player) {
				count++
				r += d[0] * sign
				c += d[1] * sign
			}
		}
		if count >= WinLength {
			return true
		}
	}
	return false
}

// WinnerAt returns the player that owns the winning line through row, col,
// or 0 when the move does not win.
func WinnerAt(b Board, row, col int) Player {
	if b.At(row, col) == Empty {
		return 0
	}
	player := Player(b[row][col])
	if WinsAt(b, row, col, player) {
		return player
	}
	return 0
}
