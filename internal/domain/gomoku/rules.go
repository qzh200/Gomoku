package gomoku

const WinLength = 5

var directions = [4][2]int{
	{0, 1},
	{1, 0},
	{1, 1},
	{1, -1},
}

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

func WinningLine(b Board, player Player) [][2]int {
	for row := range BoardSize {
		for col := range BoardSize {
			if b[row][col] != int(player) {
				continue
			}
			for _, d := range directions {
				if line := lineThrough(b, row, col, d, player); line != nil {
					return line
				}
			}
		}
	}
	return nil
}

func lineThrough(b Board, row, col int, d [2]int, player Player) [][2]int {
	if b[row][col] != int(player) {
		return nil
	}
	start := row - d[0]*(WinLength-1)
	startCol := col - d[1]*(WinLength-1)
	if !InBounds(start, startCol) {
		return nil
	}
	line := make([][2]int, 0, WinLength)
	for i := range WinLength {
		r, c := start+d[0]*i, startCol+d[1]*i
		if b[r][c] != int(player) {
			return nil
		}
		line = append(line, [2]int{r, c})
	}
	return line
}

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
