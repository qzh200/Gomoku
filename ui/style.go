package ui

import (
	mygo "github.com/egoist/mygo/ui"
	"github.com/lobehub/lobe-mygo/pkg/display"
	"github.com/lobehub/lobe-mygo/pkg/theme"
)

// palette follows the warm, paper-like surfaces of the Onevra component
// gallery: an off-white canvas, raised white cards, hairline warm borders and
// a terracotta accent.
type palette struct {
	canvas       mygo.Color
	surface      mygo.Color
	surfaceAlt   mygo.Color
	border       mygo.Color
	borderStrong mygo.Color
	text         mygo.Color
	muted        mygo.Color
	faint        mygo.Color
	accent       mygo.Color
	accentSoft   mygo.Color
	success      mygo.Color
	warning      mygo.Color
	danger       mygo.Color
}

func paletteOf(tk *theme.Tokens) palette {
	if tk != nil && tk.Dark {
		return palette{
			canvas:       mygo.Hex("#171512"),
			surface:      mygo.Hex("#1f1c18"),
			surfaceAlt:   mygo.Hex("#2a2520"),
			border:       mygo.Hex("#322d27"),
			borderStrong: mygo.Hex("#443d34"),
			text:         mygo.Hex("#f2ede4"),
			muted:        mygo.Hex("#a89f93"),
			faint:        mygo.Hex("#7b7368"),
			accent:       mygo.Hex("#d47a52"),
			accentSoft:   mygo.Hex("#33221b"),
			success:      mygo.Hex("#7aa87f"),
			warning:      mygo.Hex("#d0a25c"),
			danger:       mygo.Hex("#d97b6d"),
		}
	}
	return palette{
		canvas:       mygo.Hex("#f5f2ed"),
		surface:      mygo.Hex("#fdfcfa"),
		surfaceAlt:   mygo.Hex("#ece7df"),
		border:       mygo.Hex("#e3ddd3"),
		borderStrong: mygo.Hex("#cfc7ba"),
		text:         mygo.Hex("#262320"),
		muted:        mygo.Hex("#6e675e"),
		faint:        mygo.Hex("#9c9488"),
		accent:       mygo.Hex("#bf6640"),
		accentSoft:   mygo.Hex("#f2e2d8"),
		success:      mygo.Hex("#4a7d52"),
		warning:      mygo.Hex("#a97a2f"),
		danger:       mygo.Hex("#a8463c"),
	}
}

const (
	panelRadius = 13.0
	controlRad  = 9.0
)

// panel is one raised card. Callers add their own column inside it.
func panel(c *mygo.Context) mygo.Element {
	p := paletteOf(theme.TokensOf(c))
	return mygo.Box(c).Padding(16).Radius(panelRadius).
		Background(p.surface).Border(1, p.border)
}

// groupLabel is the small muted heading above a group of controls.
func groupLabel(c *mygo.Context, text string) {
	p := paletteOf(theme.TokensOf(c))
	mygo.Text(c, text).FontSize(14).FontWeight(600).TextColor(p.muted).LetterSpacing(0.2)
}

// fieldLabel names an individual control.
func fieldLabel(c *mygo.Context, text string) {
	p := paletteOf(theme.TokensOf(c))
	mygo.Text(c, text).FontSize(13).FontWeight(500).TextColor(p.faint)
}

// segmented is the pill switch of the reference: a soft track whose chosen
// segment lifts onto the card surface.
func segmented(c *mygo.Context, selected *int, options []string) {
	p := paletteOf(theme.TokensOf(c))
	parts := mygo.SegmentedBase(c, selected, len(options))
	track := parts.Track.Height(38).Padding(3).Gap(3).Radius(controlRad).
		Background(p.surfaceAlt).AlignSelf(mygo.Stretch)
	if selected == nil {
		return
	}
	track.Children(func() {
		for i, option := range options {
			seg := parts.Segment(i).Height(32).PaddingX(6).Radius(8).
				Cursor(mygo.CursorPointer).Grow(1).Basis(0)
			switch {
			case i == *selected:
				seg.Background(p.surface).TextColor(p.text).
					Shadow(0, 1, 2, 0, p.borderStrong)
			case seg.Hovered():
				seg.TextColor(p.text)
			default:
				seg.TextColor(p.muted)
			}
			seg.Children(func() {
				mygo.Text(c, option).FontSize(14).FontWeight(500).SingleLine()
			})
		}
	})
}

// primaryButton is the filled accent call to action.
func primaryButton(c *mygo.Context, icon *mygo.SVG, label string) mygo.Element {
	p := paletteOf(theme.TokensOf(c))
	b := mygo.ButtonBase(c).Height(42).PaddingX(16).Radius(controlRad).Cursor(mygo.CursorPointer)
	bg := p.accent
	if b.Hovered() {
		bg = p.accent.Mix(mygo.RGB(255, 255, 255), 0.12)
	}
	if b.Pressed() {
		bg = p.accent.Mix(mygo.RGB(0, 0, 0), 0.12)
	}
	b.Background(bg).TextColor(mygo.RGB(255, 255, 255))
	b.Children(func() {
		buttonContent(c, icon, label, mygo.RGB(255, 255, 255))
	})
	return b
}

// ghostButton is the bordered card-colored secondary action.
func ghostButton(c *mygo.Context, icon *mygo.SVG, label string) mygo.Element {
	p := paletteOf(theme.TokensOf(c))
	b := mygo.ButtonBase(c).Height(40).PaddingX(14).Radius(controlRad).Cursor(mygo.CursorPointer)
	line := p.border
	if b.Hovered() {
		line = p.borderStrong
	}
	b.Background(p.surface).Border(1, line).TextColor(p.text)
	b.Children(func() {
		buttonContent(c, icon, label, p.text)
	})
	return b
}

func buttonContent(c *mygo.Context, icon *mygo.SVG, label string, color mygo.Color) {
	mygo.Row(c).Gap(7).AlignItems(mygo.Center).Children(func() {
		display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
		mygo.Text(c, label).FontSize(15).FontWeight(600).TextColor(color).SingleLine()
	})
}

// chip is a soft pill with a colored status dot.
func chip(c *mygo.Context, text string, dot mygo.Color) {
	p := paletteOf(theme.TokensOf(c))
	mygo.Row(c).Gap(8).Height(32).Padding(0, 12).Radius(16).
		Background(p.surface).Border(1, p.border).AlignItems(mygo.Center).Children(func() {
		mygo.Box(c).Size(6, 6).Radius(3).Background(dot).Shrink(0)
		mygo.Text(c, text).FontSize(14).FontWeight(500).TextColor(p.text).SingleLine()
	})
}

// noticeRow is a tinted inline message with a leading icon.
func noticeRow(c *mygo.Context, icon *mygo.SVG, text string, color mygo.Color) {
	row := mygo.Row(c).Gap(9).Padding(9, 11).Radius(9).AlignItems(mygo.Center)
	row.Background(color.Alpha(0.10)).Border(1, color.Alpha(0.24)).TextColor(color)
	row.Children(func() {
		display.Icon(c, icon, display.IconProps{Size: display.IconSmall})
		mygo.Text(c, text).FontSize(14).FontWeight(500).TextColor(color)
	})
}

// tinyTag is the result badge between records.
func tinyTag(c *mygo.Context, text string, color mygo.Color) {
	mygo.Row(c).Padding(2, 8).Radius(999).Background(color.Alpha(0.12)).
		Border(1, color.Alpha(0.25)).Children(func() {
		mygo.Text(c, text).FontSize(13).FontWeight(600).TextColor(color).SingleLine()
	})
}

func tagColor(c display.TagColor) mygo.Color {
	p := paletteOf(theme.Light())
	switch c {
	case display.TagBlue:
		return p.accent
	case display.TagGreen:
		return p.success
	case display.TagGold:
		return p.warning
	case display.TagGray:
		return p.muted
	}
	return p.faint
}

// emptyHint fills a list that has nothing in it yet.
func emptyHint(c *mygo.Context, text string) {
	p := paletteOf(theme.TokensOf(c))
	box := mygo.Box(c).Padding(14, 12).Radius(9).Border(1, p.border).AlignItems(mygo.Center)
	box.Children(func() {
		mygo.Text(c, text).FontSize(14).TextColor(p.faint)
	})
}

// stoneColors returns the fill gradient ends of a stone for the given player.
func stoneColors(playerIsBlack bool, tk *theme.Tokens) (light, dark mygo.Color) {
	if tk != nil && tk.Dark {
		if playerIsBlack {
			return mygo.Hex("#5a5148"), mygo.Hex("#171310")
		}
		return mygo.Hex("#f6f0e6"), mygo.Hex("#cfc5b6")
	}
	if playerIsBlack {
		return mygo.Hex("#544b42"), mygo.Hex("#171310")
	}
	return mygo.Hex("#ffffff"), mygo.Hex("#d5ccbf")
}
