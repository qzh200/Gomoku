package ui

import (
	mygo "github.com/egoist/mygo/ui"
	"github.com/lobehub/lobe-mygo/pkg/display"
)

// Palette of awesomemygo.dev: near-black layout, blue-slate panels, hairline
// borders and a cyan accent.
var (
	colBg    = mygo.Hex("#0b0d10")
	colPanel = mygo.Hex("#12161b")
	colWell  = mygo.Hex("#0f1216")
	colLine1 = mygo.Hex("#1c2229")
	colLine2 = mygo.Hex("#252c35")
	colLine3 = mygo.Hex("#313a45")
	colFg    = mygo.Hex("#eef2f5")
	colFg2   = mygo.Hex("#a3afba")
	colFg3   = mygo.Hex("#7e8b97")
	colCyan  = mygo.Hex("#3fd3f5")
	colCyan2 = mygo.Hex("#8fe6fa")
	colGreen = mygo.Hex("#9be38a")
	colAmber = mygo.Hex("#ffb547")
	colErr   = mygo.Hex("#ff8a80")
)

const (
	panelRadius  = 14.0
	controlRad   = 9.0
	toolbarHair  = 1.0
	labelSizeEmi = 10.0
)

func paintBackdrop(page mygo.Element) {
	page.Draw(func(p *mygo.Painter, r mygo.Rect) {
		p.Fill(r, colBg, 0)
		p.FillRadialGradient(r, mygo.RadialGradient{
			Stops:  []mygo.GradientStop{{Color: colCyan.Alpha(0.10), At: 0}, {Color: colCyan.Alpha(0), At: 1}},
			Center: mygo.Point{X: 0.82, Y: 0.04},
			Radius: r.W * 0.5,
		}, 0)
		p.FillRadialGradient(r, mygo.RadialGradient{
			Stops:  []mygo.GradientStop{{Color: colAmber.Alpha(0.07), At: 0}, {Color: colAmber.Alpha(0), At: 1}},
			Center: mygo.Point{X: 0.08, Y: 0.94},
			Radius: r.W * 0.45,
		}, 0)
		p.FillRadialGradient(r, mygo.RadialGradient{
			Stops:  []mygo.GradientStop{{Color: colBg.Alpha(0), At: 0.5}, {Color: colBg.Alpha(0.9), At: 1}},
			Center: mygo.Point{X: 0.5, Y: 0.4},
			Radius: r.W * 0.62,
		}, 0)
	})
}

// panel is one framed surface. Callers add their own column inside it.
func panel(c *mygo.Context) mygo.Element {
	return mygo.Box(c).Padding(16).Radius(panelRadius).
		Background(colPanel).Border(1, colLine2)
}

// sectionHeader is the "01 / 新对局  NEW GAME" line above a panel's content.
func sectionHeader(c *mygo.Context, number, title, tag string) {
	mygo.Row(c).Gap(7).AlignItems(mygo.Center).Children(func() {
		mygo.Text(c, number).FontSize(10).FontWeight(600).TextColor(colCyan).LetterSpacing(0.6)
		mygo.Text(c, "/").FontSize(10).FontWeight(500).TextColor(colLine3)
		mygo.Text(c, title).FontSize(13).FontWeight(600).TextColor(colFg)
		mygo.Spacer(c).Grow(1).Basis(0)
		if tag != "" {
			mygo.Text(c, tag).FontSize(labelSizeEmi).FontWeight(600).TextColor(colFg3).LetterSpacing(0.9)
		}
	})
}

// fieldLabel names a control in the small muted caps of the reference.
func fieldLabel(c *mygo.Context, text string) {
	mygo.Text(c, text).FontSize(11).FontWeight(500).TextColor(colFg3).LetterSpacing(0.3)
}

// chip is a pill of muted surface with a colored status dot.
func chip(c *mygo.Context, text string, dot mygo.Color) {
	mygo.Row(c).Gap(8).Height(28).Padding(0, 11).Radius(14).
		Background(colPanel).Border(1, colLine2).AlignItems(mygo.Center).Children(func() {
		mygo.Box(c).Size(6, 6).Radius(3).Background(dot).Shrink(0)
		mygo.Text(c, text).FontSize(12).FontWeight(500).TextColor(colFg).SingleLine()
	})
}

// segmented is the site's pill switch: a well track whose chosen segment is
// filled with the foreground color.
func segmented(c *mygo.Context, selected *int, options []string) {
	parts := mygo.SegmentedBase(c, selected, len(options))
	track := parts.Track.Height(32).Padding(3).Gap(4).Radius(controlRad).
		Background(colWell).Border(1, colLine1).AlignSelf(mygo.Stretch)
	if selected == nil {
		return
	}
	track.Children(func() {
		for i, option := range options {
			seg := parts.Segment(i).Height(26).PaddingX(9).Radius(6).
				Cursor(mygo.CursorPointer).Grow(1).Basis(0)
			switch {
			case i == *selected:
				seg.Background(colFg).TextColor(colBg)
			case seg.Hovered():
				seg.Background(colLine2).TextColor(colFg)
			default:
				seg.TextColor(colFg2)
			}
			seg.Children(func() {
				mygo.Text(c, option).FontSize(12).FontWeight(500).SingleLine()
			})
		}
	})
}

// primaryButton is the filled cyan call to action.
func primaryButton(c *mygo.Context, icon *mygo.SVG, label string) mygo.Element {
	b := mygo.ButtonBase(c).Height(36).PaddingX(14).Radius(controlRad).Cursor(mygo.CursorPointer)
	bg := colCyan
	switch {
	case b.Pressed():
		bg = mygo.Hex("#2ab6d6")
	case b.Hovered():
		bg = colCyan2
	}
	b.Background(bg).TextColor(colBg)
	b.Children(func() {
		buttonContent(c, icon, label, colBg)
	})
	return b
}

// ghostButton is the bordered well button used for secondary actions.
func ghostButton(c *mygo.Context, icon *mygo.SVG, label string) mygo.Element {
	b := mygo.ButtonBase(c).Height(34).PaddingX(12).Radius(controlRad).Cursor(mygo.CursorPointer)
	line := colLine2
	switch {
	case b.Hovered():
		line = colFg3
		b.Background(colWell)
	case b.Pressed():
		line = colFg3
	}
	b.Background(colWell).Border(1, line).TextColor(colFg)
	b.Children(func() {
		buttonContent(c, icon, label, colFg)
	})
	return b
}

func buttonContent(c *mygo.Context, icon *mygo.SVG, label string, color mygo.Color) {
	mygo.Row(c).Gap(7).AlignItems(mygo.Center).Children(func() {
		display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
		mygo.Text(c, label).FontSize(13).FontWeight(600).TextColor(color).SingleLine()
	})
}

// noticeRow is a tinted inline message with a leading accent icon.
func noticeRow(c *mygo.Context, icon *mygo.SVG, text string, color mygo.Color) {
	row := mygo.Row(c).Gap(9).Padding(9, 10).Radius(10).AlignItems(mygo.Center)
	row.Background(color.Alpha(0.08)).Border(1, color.Alpha(0.22))
	row.Children(func() {
		display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
		mygo.Text(c, text).FontSize(12).FontWeight(500).TextColor(color)
	})
}

// tinyTag is the result badge between records.
func tinyTag(c *mygo.Context, text string, color mygo.Color) {
	mygo.Row(c).Padding(2, 8).Radius(999).Background(color.Alpha(0.10)).
		Border(1, color.Alpha(0.22)).Children(func() {
		mygo.Text(c, text).FontSize(11).FontWeight(600).TextColor(color).SingleLine()
	})
}

func tagColor(c display.TagColor) mygo.Color {
	switch c {
	case display.TagBlue:
		return colCyan2
	case display.TagGreen:
		return colGreen
	case display.TagGold:
		return colAmber
	case display.TagGray:
		return colFg2
	}
	return colFg3
}

// emptyHint fills a list that has nothing in it yet.
func emptyHint(c *mygo.Context, text string) {
	box := mygo.Box(c).Padding(14, 12).Radius(9).Border(1, colLine1).AlignItems(mygo.Center)
	box.Children(func() {
		mygo.Text(c, text).FontSize(12).TextColor(colFg3)
	})
}
