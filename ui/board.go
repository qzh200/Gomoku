// Package ui is the desktop interface of MyGo. It draws the board with the
// MyGo native painter and reads the game state through the application
// services.
package ui

import (
	"math"

	mygo "github.com/egoist/mygo/ui"
	"github.com/lobehub/lobe-mygo/pkg/theme"

	"mygo/internal/domain/gomoku"
)

// Board geometry. The board element is exactly boardSize wide and high, so
// the pointer position can be mapped to a cell without a layout pass.
const (
	boardCell   = 36.0
	boardMargin = 28.0
	boardSize   = boardMargin*2 + 14*boardCell
	boardRadius = 8.0
	starRadius  = 3.5
	stoneRadius = 15.0
)

// starSpots are the indices of the star points of a 15x15 board.
var starSpots = [5][2]int{{3, 3}, {3, 11}, {7, 7}, {11, 3}, {11, 11}}

// BoardView draws the board and reports the cell the user picked.
type BoardView struct {
	// Game is the snapshot to draw.
	Game *gomoku.Game
	// HoverRow and HoverCol are the cell under the pointer, -1 when none.
	HoverRow int
	HoverCol int
	// OnPick is called with a cell when the user clicks it.
	OnPick func(row, col int)

	wood  mygo.Color
	line  mygo.Color
	black mygo.Color
	white mygo.Color
	star  mygo.Color
	edge  mygo.Color
}

// newBoardView returns a board view.
func newBoardView() *BoardView {
	return &BoardView{HoverRow: -1, HoverCol: -1}
}

// applyTheme takes the board colors from the Lobe theme tokens.
func (b *BoardView) applyTheme(tk *theme.Tokens) {
	b.wood = mygo.RGB(214, 178, 128)
	b.line = mygo.RGB(120, 88, 52)
	b.black = mygo.RGB(28, 28, 32)
	b.white = mygo.RGB(248, 248, 250)
	b.star = mygo.RGB(120, 88, 52)
	b.edge = mygo.RGB(150, 110, 66)
	_ = tk
}

// View builds the board element. It draws the board and takes the clicks.
func (b *BoardView) View(c *mygo.Context) mygo.Element {
	board := mygo.Box(c).Size(boardSize, boardSize).Radius(boardRadius)
	board.HandleInput(func(ev mygo.InputEvent) bool {
		switch ev.Kind {
		case mygo.InputPointerDown:
			row, col := b.cellAt(ev.X, ev.Y)
			if row >= 0 && b.OnPick != nil {
				b.OnPick(row, col)
			}
			return true
		case mygo.InputPointerMove:
			b.HoverRow, b.HoverCol = b.cellAt(ev.X, ev.Y)
		}
		return false
	})
	board.Draw(func(p *mygo.Painter, r mygo.Rect) {
		b.paint(p, r)
	})
	return board
}

// cellAt maps a pointer position to a cell of the board. It returns -1 when
// the position is outside the grid.
func (b *BoardView) cellAt(x, y float32) (int, int) {
	if x < 0 || y < 0 {
		return -1, -1
	}
	col := int(math.Round(float64((x - boardMargin) / boardCell)))
	row := int(math.Round(float64((y - boardMargin) / boardCell)))
	if row < 0 || row > 14 || col < 0 || col > 14 {
		return -1, -1
	}
	return row, col
}

// paint draws the board, the grid, the stones and the hovered cell.
func (b *BoardView) paint(p *mygo.Painter, r mygo.Rect) {
	p.Fill(r, b.wood, boardRadius)
	p.Stroke(r, b.edge, boardRadius, 1)

	first := r.X + boardMargin
	top := r.Y + boardMargin
	for i := range 15 {
		offset := float32(i) * boardCell
		p.Line(first, top+offset, first+14*boardCell, top+offset, 1, b.line)
		p.Line(first+offset, top, first+offset, top+14*boardCell, 1, b.line)
	}
	for _, spot := range starSpots {
		p.FillPath((&mygo.Path{}).Circle(first+float32(spot[1])*boardCell, top+float32(spot[0])*boardCell, starRadius), b.star)
	}

	if b.Game == nil {
		return
	}
	for row := range 15 {
		for col := range 15 {
			cell := b.Game.Board[row][col]
			if cell == 0 {
				continue
			}
			b.paintStone(p, r, row, col, cell, false)
		}
	}
	if b.HoverCellFree() {
		b.paintStone(p, r, b.HoverRow, b.HoverCol, int(b.Game.Current), true)
	}
}

// HoverCellFree reports whether the hovered cell can take a stone.
func (b *BoardView) HoverCellFree() bool {
	if b.Game == nil || b.HoverRow < 0 || b.HoverCol < 0 {
		return false
	}
	return b.Game.Board[b.HoverRow][b.HoverCol] == 0
}

// paintStone draws one stone, or its ghost while the pointer hovers.
func (b *BoardView) paintStone(p *mygo.Painter, r mygo.Rect, row, col, cell int, ghost bool) {
	cx := r.X + boardMargin + float32(col)*boardCell
	cy := r.Y + boardMargin + float32(row)*boardCell
	fill := b.black
	if cell == 2 {
		fill = b.white
	}
	if ghost {
		fill = fill.Alpha(0.35)
	}
	p.FillPath((&mygo.Path{}).Circle(cx, cy, stoneRadius), fill)
	if cell == 2 {
		p.StrokePath((&mygo.Path{}).Circle(cx, cy, stoneRadius), 1, mygo.RGB(180, 180, 186))
	}
}

// stoneColor returns the color of a stone of the given player.
func stoneColor(player gomoku.Player) mygo.Color {
	if player == gomoku.PlayerBlack {
		return mygo.RGB(28, 28, 32)
	}
	return mygo.RGB(248, 248, 250)
}
