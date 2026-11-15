package ui

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	mygoui "github.com/egoist/mygo/ui"
)

func writeSnapshot(t *testing.T, tester *mygoui.Tester, name string) {
	t.Helper()
	path := filepath.Join(os.TempDir(), name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, tester.Image()); err != nil {
		t.Fatal(err)
	}
	t.Log(path)
}

func TestSnapshotEmpty(t *testing.T) {
	app := newTestApp(t)
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)
	writeSnapshot(t, tester, "mygo-empty.png")
}

func TestSnapshotGame(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()
	for _, move := range [][2]int{{7, 7}, {7, 8}, {6, 6}, {8, 8}, {5, 5}, {5, 6}, {4, 4}, {4, 5}, {3, 3}, {3, 4}} {
		app.pick(move[0], move[1])
	}
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)
	writeSnapshot(t, tester, "mygo-game.png")
}

func TestSnapshotDark(t *testing.T) {
	app := newTestApp(t)
	app.newGame()
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.SetDark(true)
	tester.Advance(200 * time.Millisecond)
	writeSnapshot(t, tester, "mygo-dark.png")
}

func TestSnapshotSmall(t *testing.T) {
	app := newTestApp(t)
	app.newGame()
	tester := mygoui.NewTester(app.View(), 980, 660)
	tester.Advance(200 * time.Millisecond)
	writeSnapshot(t, tester, "mygo-small.png")
}

func TestSnapshotFinished(t *testing.T) {
	app := newTestApp(t)
	app.st.modeIndex = 2
	app.newGame()
	for _, move := range [][2]int{{7, 7}, {7, 8}, {6, 6}, {8, 8}, {5, 5}, {5, 6}, {4, 4}, {4, 5}, {3, 3}} {
		app.pick(move[0], move[1])
	}
	tester := mygoui.NewTester(app.View(), 1180, 800)
	tester.Advance(200 * time.Millisecond)
	writeSnapshot(t, tester, "mygo-finished.png")
}
