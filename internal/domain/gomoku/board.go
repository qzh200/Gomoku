package gomoku

const BoardSize = 15

const (
	Empty = 0
	Black = 1
	White = 2
)

type Player int

const (
	PlayerBlack Player = Black
	PlayerWhite Player = White
)

func (p Player) Other() Player {
	if p == PlayerBlack {
		return PlayerWhite
	}
	return PlayerBlack
}

func (p Player) String() string {
	if p == PlayerBlack {
		return "black"
	}
	return "white"
}

type Board [BoardSize][BoardSize]int

func NewBoard() Board {
	return Board{}
}

func InBounds(row, col int) bool {
	return row >= 0 && row < BoardSize && col >= 0 && col < BoardSize
}

func (b Board) At(row, col int) int {
	if !InBounds(row, col) {
		return Empty
	}
	return b[row][col]
}

func (b Board) IsEmpty(row, col int) bool {
	return b.At(row, col) == Empty
}

func (b *Board) Place(row, col int, player Player) bool {
	if !InBounds(row, col) || b[row][col] != Empty {
		return false
	}
	b[row][col] = int(player)
	return true
}

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

func (b Board) Cells() [][]int {
	cells := make([][]int, BoardSize)
	for row := range cells {
		cells[row] = make([]int, BoardSize)
		copy(cells[row], b[row][:])
	}
	return cells
}
