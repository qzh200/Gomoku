// Package gomoku holds the game rules and the domain state of a Gomoku game.
// It must not depend on HTTP, storage or UI packages.
package gomoku

// BoardSize is the number of lines and columns of the board.
const BoardSize = 15

// Cell values used by the board.
const (
	Empty = 0
	Black = 1
	White = 2
)

// Player is the color of a stone.
type Player int

// The two players.
const (
	PlayerBlack Player = Black
	PlayerWhite Player = White
)

// Other returns the opponent of p.
func (p Player) Other() Player {
	if p == PlayerBlack {
		return PlayerWhite
	}
	return PlayerBlack
}

// String names the player.
func (p Player) String() string {
	if p == PlayerBlack {
		return "black"
	}
	return "white"
}

// Board is a 15x15 grid of cells. The zero value is an empty board.
type Board [BoardSize][BoardSize]int

// NewBoard returns an empty board.
func NewBoard() Board {
	return Board{}
}

// InBounds reports whether row and col address a cell of the board.
func InBounds(row, col int) bool {
	return row >= 0 && row < BoardSize && col >= 0 && col < BoardSize
}

// At returns the cell at row, col, or Empty when out of bounds.
func (b Board) At(row, col int) int {
	if !InBounds(row, col) {
		return Empty
	}
	return b[row][col]
}

// IsEmpty reports whether the cell at row, col is free.
func (b Board) IsEmpty(row, col int) bool {
	return b.At(row, col) == Empty
}

// Place writes a stone. It reports whether the cell was free and in bounds.
func (b *Board) Place(row, col int, player Player) bool {
	if !InBounds(row, col) || b[row][col] != Empty {
		return false
	}
	b[row][col] = int(player)
	return true
}

// IsFull reports whether the board holds no free cell.
func (b Board) IsFull() bool {
	for row := range b {
		for col := range b {
			if b[row][col] == Empty {
				return false
			}
		}
	}
	return true
}

// Cells returns the board as a plain slice of slices, row major.
func (b Board) Cells() [][]int {
	cells := make([][]int, BoardSize)
	for row := range cells {
		cells[row] = make([]int, BoardSize)
		copy(cells[row], b[row][:])
	}
	return cells
}
