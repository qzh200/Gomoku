// Package assets holds binary assets shared by the window chrome and the
// in-app header, so both show the same icon.
package assets

import _ "embed"

// IconPNG is the application icon, shown in the title bar and in the header.
//
//go:embed icon.png
var IconPNG []byte
