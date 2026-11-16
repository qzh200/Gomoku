package ui

import (
	"math"

	mygo "github.com/egoist/mygo/ui"
	"github.com/lobehub/lobe-mygo/pkg/theme"

	"mygo/internal/domain/gomoku"
)

const (
	boardGutter  = 24.0
	boardMinSize = 420.0
	boardMaxSize = 760.0
	boardRadius  = 10.0
	labelSize    = 11.0
	stoneRatio   = 0.42
	starRatio    = 0.10
	shadowOffset = 1.5
	shadowRadius = 3.0
)

var columnLabels = []string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"}

var rowLabels = []string{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"}

var starSpots = [5][2]int{{3, 3}, {3, 11}, {7, 7}, {11, 3}, {11, 11}}

type geometry struct {
	Size float32

	Cell float32

	OriginX float32
	OriginY float32
}

type BoardView struct {
	Game *gomoku.Game

	HoverRow int
	HoverCol int

	OnPick func(row, col int)

	wood     mygo.Color
	line     mygo.Color
	black    mygo.Color
	blackTop mygo.Color
	white    mygo.Color
	whiteTop mygo.Color
	star     mygo.Color
	edge     mygo.Color
	label    mygo.Color
	accent   mygo.Color
	labelW   map[string]float32
	labelH   float32

	rect mygo.Rect
}

func newBoardView() *BoardView {
	return &BoardView{HoverRow: -1, HoverCol: -1, labelW: map[string]float32{}}
}

func (b *BoardView) applyTheme(tk *theme.Tokens) {
	b.wood = mygo.RGB(214, 178, 128)
	b.line = mygo.RGB(120, 88, 52)
	b.black = mygo.RGB(16, 16, 20)
	b.blackTop = mygo.RGB(74, 74, 84)
	b.white = mygo.RGB(214, 214, 222)
	b.whiteTop = mygo.RGB(255, 255, 255)
	b.star = mygo.RGB(120, 88, 52)
	b.edge = mygo.RGB(150, 110, 66)
	b.label = mygo.RGB(110, 78, 44)
	b.accent = mygo.RGB(214, 96, 77)
	if tk != nil {
		b.accent = tk.Error
	}
}

func (b *BoardView) View(c *mygo.Context, size float32) mygo.Element {
	geo := geometryOf(size)
	b.measureLabels(c)
	board := mygo.Box(c).Size(geo.Size, geo.Size).Radius(boardRadius)
	board.HandleInput(func(ev mygo.InputEvent) bool {
		switch ev.Kind {
		case mygo.InputPointerDown:
			row, col := b.cellAt(geo, ev.X, ev.Y)
			if row >= 0 && b.OnPick != nil {
				b.OnPick(row, col)
			}
			return true
		case mygo.InputPointerMove:
			b.HoverRow, b.HoverCol = b.cellAt(geo, ev.X, ev.Y)
		}
		return false
	})
	board.Draw(func(p *mygo.Painter, r mygo.Rect) {
		b.paint(p, r, geo)
	})
	return board
}

func (b *BoardView) measureLabels(c *mygo.Context) {
	for _, label := range columnLabels {
		if _, ok := b.labelW[label]; ok {
			continue
		}
		width, height := c.MeasureText(0, mygo.Span{Text: label, Size: labelSize})
		b.labelW[label] = width
		b.labelH = height
	}
}

func geometryOf(size float32) geometry {
	grid := size - 2*boardGutter
	cell := grid / 14
	return geometry{
		Size:    size,
		Cell:    cell,
		OriginX: boardGutter,
		OriginY: boardGutter,
	}
}

func (b *BoardView) cellAt(geo geometry, x, y float32) (int, int) {
	if x < 0 || y < 0 || x > geo.Size || y > geo.Size {
		return -1, -1
	}
	col := int(math.Round(float64((x - geo.OriginX) / geo.Cell)))
	row := int(math.Round(float64((y - geo.OriginY) / geo.Cell)))
	if row < 0 || row > 14 || col < 0 || col > 14 {
		return -1, -1
	}
	return row, col
}

func (b *BoardView) paint(p *mygo.Painter, r mygo.Rect, geo geometry) {
	b.rect = r
	p.Fill(r, b.wood, boardRadius)
	p.Stroke(r, b.edge, boardRadius, 1)

	first := r.X + geo.OriginX
	top := r.Y + geo.OriginY
	span := geo.Cell * 14
	for i := range 15 {
		offset := float32(i) * geo.Cell
		p.Line(first, top+offset, first+span, top+offset, 1, b.line)
		p.Line(first+offset, top, first+offset, top+span, 1, b.line)
	}
	for _, spot := range starSpots {
		p.FillPath((&mygo.Path{}).Circle(first+float32(spot[1])*geo.Cell, top+float32(spot[0])*geo.Cell, geo.Cell*starRatio), b.star)
	}

	b.paintLabels(p, r, geo)

	if b.Game == nil {
		return
	}
	b.paintWinningLine(p, r, geo)
	last := b.Game.LastMove
	for row := range 15 {
		for col := range 15 {
			cell := b.Game.Board[row][col]
			if cell == 0 {
				continue
			}
			b.paintStone(p, r, geo, row, col, cell, false)
			if last != nil && last.Row == row && last.Col == col {
				b.paintLastMove(p, r, geo, row, col)
			}
		}
	}
	if b.hoverCellFree() {
		b.paintStone(p, r, geo, b.HoverRow, b.HoverCol, int(b.Game.Current), true)
	}
}

func (b *BoardView) hoverCellFree() bool {
	if b.Game == nil || b.HoverRow < 0 || b.HoverCol < 0 {
		return false
	}
	return b.Game.Board[b.HoverRow][b.HoverCol] == 0
}

func (b *BoardView) paintLabels(p *mygo.Painter, r mygo.Rect, geo geometry) {
	first := r.X + geo.OriginX
	top := r.Y + geo.OriginY
	for i, label := range columnLabels {
		x := first + float32(i)*geo.Cell
		b.paintLabel(p, label, x-b.labelW[label]/2, r.Y+4, b.label)
	}
	for i, label := range rowLabels {
		y := top + float32(i)*geo.Cell
		b.paintLabel(p, label, r.X+4, y-b.labelH/2, b.label)
	}
}

func (b *BoardView) paintLabel(p *mygo.Painter, text string, x, y float32, color mygo.Color) {
	p.Text(x, y, text, labelSize, color)
}

func (b *BoardView) paintWinningLine(p *mygo.Painter, r mygo.Rect, geo geometry) {
	if b.Game.Status != gomoku.StatusFinished || b.Game.Winner == nil {
		return
	}
	line := gomoku.WinningLine(b.Game.Board, *b.Game.Winner)
	if len(line) < 2 {
		return
	}

	radius := geo.Cell * stoneRatio
	for i := 1; i < len(line); i++ {
		from := mygo.Point{
			X: r.X + geo.OriginX + float32(line[i-1][1])*geo.Cell,
			Y: r.Y + geo.OriginY + float32(line[i-1][0])*geo.Cell,
		}
		to := mygo.Point{
			X: r.X + geo.OriginX + float32(line[i][1])*geo.Cell,
			Y: r.Y + geo.OriginY + float32(line[i][0])*geo.Cell,
		}
		dx, dy := to.X-from.X, to.Y-from.Y
		length := float32(math.Hypot(float64(dx), float64(dy)))
		if length == 0 {
			continue
		}
		inset := radius * 0.9
		p.StrokePolyline([][2]float32{
			{from.X + dx/length*inset, from.Y + dy/length*inset},
			{to.X - dx/length*inset, to.Y - dy/length*inset},
		}, geo.Cell*0.22, b.accent.Alpha(0.85))
	}
}

func (b *BoardView) paintLastMove(p *mygo.Painter, r mygo.Rect, geo geometry, row, col int) {
	cx := r.X + geo.OriginX + float32(col)*geo.Cell
	cy := r.Y + geo.OriginY + float32(row)*geo.Cell
	radius := geo.Cell * 0.16
	p.FillPath((&mygo.Path{}).Circle(cx, cy, radius), b.accent)
}

func (b *BoardView) paintStone(p *mygo.Painter, r mygo.Rect, geo geometry, row, col, cell int, ghost bool) {
	cx := r.X + geo.OriginX + float32(col)*geo.Cell
	cy := r.Y + geo.OriginY + float32(row)*geo.Cell
	radius := geo.Cell * stoneRatio
	light := b.blackTop
	dark := b.black
	if cell == 2 {
		light = b.whiteTop
		dark = b.white
	}
	if ghost {
		box := mygo.Rect{X: cx - radius, Y: cy - radius, W: radius * 2, H: radius * 2}
		p.Fill(box, b.accent.Alpha(0.18), radius)
		p.Stroke(box, b.accent.Alpha(0.75), radius, 1.5)
		return
	}
	shadow := mygo.Rect{X: cx - radius, Y: cy - radius + shadowOffset, W: radius * 2, H: radius * 2}
	p.Fill(shadow, mygo.RGB(120, 90, 50).Alpha(0.35), radius)
	p.Clip(shadow, radius, func() {
		p.FillRadialGradient(shadow, mygo.RadialGradient{
			Stops:  []mygo.GradientStop{{Color: light, At: 0}, {Color: dark, At: 1}},
			Center: mygo.Point{X: 0.34, Y: 0.3},
			Radius: radius * 1.6,
		}, radius)
	})
	if cell == 2 {
		p.Stroke(mygo.Rect{X: cx - radius, Y: cy - radius, W: radius * 2, H: radius * 2}, mygo.RGB(170, 170, 178), radius, 1)
	}
}

func stoneColor(player gomoku.Player) mygo.Color {
	if player == gomoku.PlayerBlack {
		return mygo.RGB(16, 16, 20)
	}
	return mygo.RGB(240, 240, 244)
}
